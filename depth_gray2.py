########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import cv2


def startZED():
    print('start zed camera')
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    
    return zed
    

def extractZED(zed):
    print('extract the zed frame')

    res = sl.Resolution()
    res.width = 720
    res.height = 404

    camera_model = zed.get_camera_information().camera_model

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    depth_image = sl.Mat(res.width, res.height)

    cv2.namedWindow("Grayscale Depth", cv2.WINDOW_NORMAL)
    zed.retrieve_image(depth_image, sl.VIEW.LEFT,sl.MEM.CPU, res)
    depth_array = depth_image.get_data()

    return depth_array

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            #viewer.updateData(point_cloud)
            zed.retrieve_image(depth_image, sl.VIEW.LEFT,sl.MEM.CPU, res)
            depth_array = depth_image.get_data()
            
            #viewer.updateData(depth_image)

            cv2.imshow("Grayscale Depth", depth_array)

    viewer.exit()
    zed.close()

def closeZED(zed):
    zed.close()


if __name__ == "__main__":
    zed = startZED()
    extractZED(zed)
    closeZED(zed)


    res = sl.Resolution()
    res.width = 720
    res.height = 404

    camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(len(sys.argv), sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    depth_image = sl.Mat(res.width, res.height)

    cv2.namedWindow("Grayscale Depth", cv2.WINDOW_NORMAL)

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            #viewer.updateData(point_cloud)
            zed.retrieve_image(depth_image, sl.VIEW.LEFT,sl.MEM.CPU, res)
            depth_array = depth_image.get_data()
            
            #viewer.updateData(depth_image)

            cv2.imshow("Grayscale Depth", depth_array)

    viewer.exit()
    zed.close()
