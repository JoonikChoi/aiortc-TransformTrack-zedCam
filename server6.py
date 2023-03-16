import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import math
import av
import numpy as np
import time

from multiprocessing import Process, Pipe, shared_memory, Lock
from fractions import Fraction

lock = Lock()

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
max_size = 1024 * 1024 * 5  # 5MB
data_shape = (404, 1440, 4)
memName = None

def extractFrame(viewer, zed, res):

    rgb_image = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_image = sl.Mat(res.width, res.height)

    if viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(rgb_image, sl.VIEW.LEFT,sl.MEM.CPU, res)
            rgb_array = rgb_image.get_data()
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH, sl.MEM.CPU, res) # 방법 2 : depth array를 8bit gray scale map으로 변환
            depth_array = depth_image.get_data()
            arr_concatenated = np.concatenate([rgb_array, depth_array], axis=1)

            return arr_concatenated
        
def startZED():
    print("Start ZED APP ... Press 'Esc' to quit")

    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 720
    res.height = 404

    camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(len(sys.argv), sys.argv, camera_model, res)

    return zed, viewer, res

def closeZED(viewer, zed):
    print("Close the ZED APP")
    viewer.exit()
    zed.close()

def sharedMem(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    return existing_shm
    


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self):
        super().__init__()  # don't forget this!
        self.shm = sharedMem(memName)
        self.arr = np.ndarray(shape=data_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.tmp_pts = 0

    async def recv(self):

    
        lock.acquire()
        zed_frame = av.VideoFrame.from_ndarray(self.arr[:], format='rgba')
        lock.release()

        #zed_frame.pts = frame.pts
        #zed_frame.time_base = frame.time_base # 요기가 원본
        zed_frame.pts = self.tmp_pts
        zed_frame.time_base = Fraction(1, 90000) # 임시적으로 pts, time_base 설정함. 나중에 수정 필요할 지도 모름.
        self.tmp_pts = self.tmp_pts + 2970

        return zed_frame 


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack()
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('server.crt', 'server.key')

def operateZED(name):
    print('Operating ZED')

    my_shm = sharedMem(name)
    arr = np.ndarray(shape=data_shape, dtype=np.uint8, buffer=my_shm.buf)
    zed, viewer, res= startZED()

    while viewer.is_available():
        brr = extractFrame(viewer, zed, res)
        lock.acquire()
        data = brr[:]
        arr[:] = data
        lock.release()

    closeZED(viewer, zed)

def appServer(args):
    #ssl_context = None
    ssl_context = ssl.SSLContext()
    ssl_context.load_cert_chain('server.crt', 'server.key')
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context,
    )

def simpleProcess(name):
    print('simple process!')
    my_shm = sharedMem(name)
    arr = np.ndarray(shape=(5,), dtype=np.uint8, buffer=my_shm.buf)

    while True:
        lock.acquire()
        print(arr[:])
        lock.release()
        time.sleep(0.001)

if __name__ == "__main__":    
    print('main start...')

    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="192.168.1.45", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create a numpy array that shares the memory block
    shm = shared_memory.SharedMemory(create=True, size=max_size)
    arr = np.ndarray(shape=data_shape, dtype=np.uint8, buffer=shm.buf)
    arr[:] = np.random.randint(0, 2, size=data_shape, dtype=np.uint8)

    memName = shm.name
    print('shared memory >>', arr[:])
    print('shared memory name?>>', memName)
    
    p = Process(target=operateZED, args=(memName, ))
    p.start()

    ## Create App ##
    ssl_context = ssl.SSLContext()
    ssl_context.load_cert_chain('server.crt', 'server.key')
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context,
    )

    print('code end-line')

