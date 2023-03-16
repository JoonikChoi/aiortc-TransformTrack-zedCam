# install & getting started - https://python-socketio.readthedocs.io/en/latest/client.html
# pip install "python-socketio[asyncio_client]==4.6.1"
# ERR - socketio.exceptions.ConnectionError: OPEN packet not returned by server
# Sol - https://stackoverflow.com/questions/66809068/python-socketio-open-packet-not-returned-by-the-server
import argparse
import asyncio
import socketio
import time
import json

import platform

import random

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.contrib.signaling import object_from_string, object_to_string
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp


import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
from multiprocessing import Process, shared_memory, Lock
from fractions import Fraction
import numpy as np
from av import VideoFrame
import av
memName = None
data_shape = (404, 1440, 4)
lock = Lock()


print("portal_webRTC_webcam_streaming")

isChannelReady = False
isStarted = False
pcs = dict()
# turnReady = False
RTCClientList = []
time_start = None

# pc = RTCPeerConnection()
config = RTCConfiguration([
        RTCIceServer("stun:stun.l.google:19302"),
        # RTCIceServer("turn:computeengineondemand.appspot.com/turn", "41784574", "4080218913"),
        # RTCIceServer("turn:3.38.108.27", "usr", "pass","password"),
])

pcConfig = {
  'iceServers': [{
    "urls": "turn:3.38.108.27",
    "username":"usr",
    "credential":"pass"
  }]
}

relay = None
webcam = None


# cmd_manager.toHome()

def channel_send(channel, message):
    print(channel, ">", message)
    channel.send(message)

def create_local_tracks(play_from, decode):
    global relay, webcam

    if play_from:
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    else:
#        options = {"framerate": "30", "video_size": "1280x720"}
        # options = {"framerate": "30", "video_size": "1920x1080", "pixel_format":"yuv420p"}
        #        options = {"framerate": "30", "video_size": "960x720"}
#        options = {"framerate": "30", "video_size": "1280x720"}
#        options = {"framerate": "30", "video_size": "1280x960"}
#        options = {"framerate": "30", "video_size": "1920x1080", "pixel_format":"rgb24"}
#        options = {"framerate": "30", "video_size": "1920x1080", "pixel_format":"yuv420p"}
#        options = {"framerate": "30", "input_format":"mjpeg", "video_size": "2592x1944"}
#        options = {"framerate": "30", "input_format":"yuv420p", "video_size": "1920x1080"}
        options = {"framerate": "30", "input_format":"mjpeg", "video_size": "640x480"}
#        options = {"framerate": "30", "video_size": "1920x1080", "c:v":"mjpeg", "i":"/dev/video0", "vcodec":"copy"}
#        options = {"framerate": "30", "video_size": "960x540"}
        if relay is None:
            if platform.system() == "Darwin":
                webcam = MediaPlayer(
                    "default:none", format="avfoundation", options=options
                )
            elif platform.system() == "Windows":
                webcam = MediaPlayer(
                    "video=Integrated Camera", format="dshow", options=options
                )
            else:
                webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
            relay = MediaRelay()
        return None, relay.subscribe(webcam.video)

def force_codec(pc, sender, forced_codec): 
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
    [codec for codec in codecs if codec.mimeType == forced_codec]
    )

async def sendMessage(dest,msg):
    packet = {
        'from': sio.sid,
        'to': dest,
        'message': msg,
        }
    await sio.emit("msg-v1",json.dumps(packet))

    # await sio.emit("msg-v1",json.dumps(packet))
    # print("send msg-v1: " + str(packet))
    return

def current_stamp():
    global time_start

    if time_start is None:
        time_start = time.time()
        return 0
    else:
        return int((time.time() - time_start) * 1000000)
    
def extractFrame(viewer, zed, res, start_time):
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    x = round(point_cloud.get_width() / 2)
    y = round(point_cloud.get_height() / 2)

    rgb_image = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_image = sl.Mat(res.width, res.height)


    if viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(rgb_image, sl.VIEW.LEFT,sl.MEM.CPU, res)
            rgb_array = rgb_image.get_data()
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH, sl.MEM.CPU, res) # 방법 2 : depth array를 8bit gray scale map으로 변환
            depth_array = depth_image.get_data()
            zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).set_nanoseconds(0)
            time_stamp= zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            #zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).set_nanoseconds(0)
            time_base = 1 / 30
            pts = (time_stamp-start_time * 1000000000) * (time_base)
            #print('get_timestamp()>', time_stamp-start_time)


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

def operateZED(name):
    print('Operating ZED')
    # Read and modify the data in the shared array
    #print(arr[:])
    my_shm = sharedMem(name)
    arr = np.ndarray(shape=data_shape, dtype=np.uint8, buffer=my_shm.buf)

    #data = arr[:]
    #data += 2
    #arr[:] = data
    zed, viewer, res= startZED()


    start_time= zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

    while viewer.is_available():
        #data = arr[:]
        #data += 2
        #arr[:] = data
        #print(arr[:])
        brr = extractFrame(viewer, zed, res, start_time)
        lock.acquire()
        data = brr[:]
        arr[:] = data
        #print(arr[:])
        lock.release()
    #print(arr[:])

    closeZED(viewer, zed)    
    
def sharedMem(name):
    existing_shm = shared_memory.SharedMemory(name=name)
    #arr = np.ndarray(shape=(max_size,), dtype=np.uint8, buffer=existing_shm.buf)
    #print('existing_shm name >>>', existing_shm.name)
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
        self.pts = 0

    async def recv(self):
        

        lock.acquire()
        #print(self.arr[:])
        zed_frame = av.VideoFrame.from_ndarray(self.arr[:], format='rgba')
        lock.release()
        
        zed_frame.pts = self.pts
        zed_frame.time_base = Fraction(1, 90000)
        self.pts = self.pts + 2970 # 임시 값, 추후 수정 필요할지도

        print(zed_frame.pts)

        return zed_frame   


async def createPeerConnection(socketId):
    try:
        # pc = RTCPeerConnection()
        pc = RTCPeerConnection(config)
        channel = pc.createDataChannel("chat")


        async def send_pings():
            while True:
                #channel_send(channel, "ping %d" % current_stamp())
                channel_send(channel, "%d %d %d" % (random.randrange(-5,6),random.randrange(-5,6),random.randrange(-5,6)))
                await asyncio.sleep(1)        

        @channel.on("open")
        def on_open():
            asyncio.ensure_future(send_pings())


        @channel.on("chat")
        def on_message(message):
            print(channel, "<", message)

            if isinstance(message, str) and message.startswith("pong"):
                elapsed_ms = (current_stamp() - int(message[5:])) / 1000
                print(" RTT %.2f ms" % elapsed_ms)        
        
        pc.addTrack(
            VideoTransformTrack()
        )


        # audio, video = create_local_tracks(
        # args.play_from, decode=not args.play_without_decoding
        # )
        # # addStream
        # if audio:
        #     audio_sender = pc.addTrack(audio)
        #     if args.audio_codec:
        #         force_codec(pc, audio_sender, args.audio_codec)
        #     elif args.play_without_decoding:
        #         raise Exception("You must specify the audio codec using --audio-codec")

        # if video:
        #     video_sender = pc.addTrack(video)
        #     if args.video_codec:
        #         force_codec(pc, video_sender, args.video_codec)
        #     elif args.play_without_decoding:
        #         raise Exception("You must specify the video codec using --video-codec")
        
        pcs[socketId]=pc

        print("Created RTCPeerConnection")
        print("socket ID: " + socketId)
        print("pcs: " + str(pcs))

        return
    except Exception as e:
        print('Failed to create PeerConnection, exception: ' + str(e))
        return

async def doCall(socketId):
    await pcs[socketId].setLocalDescription(await pcs[socketId].createOffer())
    # await sendMessage(socketId,str(pcs[socketId].localDescription))
    await sendMessage(socketId,json.dumps(
            {
                "sdp": pcs[socketId].localDescription.sdp, 
                "type": pcs[socketId].localDescription.type
                }
        ))
    # await sendMessage(socketId,"helllloooooooo")
    return


async def maybeStart(socketId):
    print(">>>>>>>maybeStart")
    await createPeerConnection(socketId)
    await doCall(socketId)
    return


# sio = socketio.AsyncClient()
sio = socketio.AsyncClient(engineio_logger=True,logger=True,ssl_verify=False)

@sio.event
async def message(data):
    print('received a message!', data)

@sio.event
async def connect():
    print("socket connected!")
    print('my sid is', sio.sid)

@sio.event
async def connect_error(data):
    print("The connection failed!")

@sio.event
async def disconnect():
    print("I'm disconnected!")



@sio.on("created")
async def created(info,room):
    print("Created room" + room)
    return

@sio.on("full")
async def full(room):
    print("Room " + room + " is full")
    return

# @sio.on("joined")       # not used at robot client (always create room)
# async def joined(room):
#     print("joined: " + room)
#     isChannelReady = true
#     return@sio.on('msg-v2')

@sio.on("msg-v1")
async def on_message1(packet):
    # message = packet.message
    # print(str(packet))
    print("msg-v1: " + str(packet))
    # message = json.loads(str(packet))
    message = packet["message"]
    sidFrom = packet['from']
    # print(message)
    try:
        if message == "connection request":
            RTCClientList.append(dict({"socketId": sidFrom}))   # structure is wired..?
            print(RTCClientList)
            await maybeStart(sidFrom)

        elif message['type']:
            print()
            print(message['type'])
            print(isinstance(message,RTCSessionDescription))
            print(isinstance(message,RTCIceCandidate))
            print()
            
            if message["type"] in ["answer", "offer"]:
                # print(str(RTCSessionDescription(**message)))
                await pcs[sidFrom].setRemoteDescription(RTCSessionDescription(**message))

                if message['type'] in ['offer']:
                    # doAnswer
                    # Infact, this program wouldn't get offer
                    await pcs[sidFrom].setLocalDescription(await pcs[sidFrom].createAnswer())
                    sendMessage(sidFrom,pcs[sidFrom].localDescription)

            elif message['type'] == 'candidate':
                candidate = candidate_from_sdp(message["candidate"].split(":", 1)[1])
                candidate.sdpMid = message["id"]
                candidate.sdpMLineIndex = message["label"]
                await pcs[sidFrom].addIceCandidate(candidate)
            else:
                print('it is wrong type')
        elif message == 'bye':
            print('Exiting')
        else:
            print('''message doesn't contain any type''')


    except Exception as e:
        print("msg-v1 exception: " + str(e))
    return

@sio.on('msg-v2')
async def on_message2(packet):
    # cmd_manager.update_commandFile(str(data.get("message")))
    return
 

async def main():
    # await sio.connect('http://localhost:5000')
    await sio.connect(url='https://192.168.1.11:3333', transports = 'websocket')
    # await sio.connect(url='https://192.168.0.11:3333',transports='websocket')
    # await sio.connect(url='https://192.168.0.2:3333',transports='websocket')

    serviceProfile = {
        'socketId':sio.sid,
        'room':'room:'+sio.sid,
        'type':'robot',
        'state': { 'roomId' : sio.sid , 'socketId' : sio.sid},        
        'description':'Streamer',
        'contents':{'stream':'{video,audio}'}
        }
    await sio.emit('Start_Service', json.dumps(serviceProfile))
    # task = sio.start_background_task(my_background_task, 123)
    await sio.wait()

print('starting socket.io-client')
async def my_background_task(my_argument):
    while True:
        # await sio.emit('echo', 'test message')
        # await sio.sleep(10)
        return
    # do some background work here!

# async def main():
#     pass

if __name__ =="__main__":
    max_size = 1024 * 1024 * 5  # 5MB

    shm = shared_memory.SharedMemory(create=True, size=max_size)
    arr = np.ndarray(shape=data_shape, dtype=np.uint8, buffer=shm.buf)
    arr[:] = np.random.randint(0, 2, size=data_shape, dtype=np.uint8)
    memName = shm.name
    print('shared memory name?>>', memName)

    
    p = Process(target=operateZED, args=(memName, ))
    p.start()


    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
