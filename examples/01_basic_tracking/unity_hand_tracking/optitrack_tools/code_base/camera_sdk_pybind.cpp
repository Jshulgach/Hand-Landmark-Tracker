#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "cameralibrary.h"

namespace py = pybind11;
using namespace CameraLibrary;

PYBIND11_MODULE(optitrack_cam, m) {
    m.doc() = "Python bindings for OptiTrack Camera Library SDK with Latency Fix";

    // Initialize SDK & Global Manager 
    m.def("initialize_sdk", []() {
        CameraLibrary_EnableDevelopment(); 
        CameraManager::X(); 
    });

    m.def("shutdown_sdk", []() {
        CameraManager::X().Shutdown(); 
    });

    // Camera Class
    py::class_<Camera>(m, "Camera")
        .def("start", &Camera::Start)
        .def("stop", &Camera::Stop, py::arg("turn_numeric_off") = true)
        .def("width", &Camera::Width)
        .def("height", &Camera::Height)
        .def("set_video_type", [](Camera &c, int type) { c.SetVideoType((Core::eVideoMode)type); })
        .def("set_aec", &Camera::SetAEC)
        .def("set_agc", &Camera::SetAGC)
        .def("set_exposure", &Camera::SetExposure)
        .def("get_exposure", &Camera::Exposure)
        .def("set_text_overlay", &Camera::SetTextOverlay)
        .def("release", &Camera::Release)
        // Standard GetFrame
        .def("get_frame", &Camera::GetFrame, py::return_value_policy::reference)
        // CUSTOM: Get Latest Frame (Drains queue to remove delay)
        .def("get_latest_frame", [](Camera &c) {
            Frame* frame = nullptr;
            Frame* latestFrame = nullptr;

            // Keep grabbing frames until the buffer is empty
            while ((frame = c.GetFrame()) != nullptr) {
                if (latestFrame) {
                    latestFrame->Release(); // Discard the older frame
                }
                latestFrame = frame;
            }
            return latestFrame; // Return only the absolute newest one
        }, py::return_value_policy::reference);

    // Frame Class 
    py::class_<Frame>(m, "Frame")
        .def("width", &Frame::Width)
        .def("height", &Frame::Height)
        .def("release", &Frame::Release)
        .def("rasterize", [](Frame &f, int width, int height) {
            std::vector<unsigned char> buffer(width * height * 4);
            Bitmap framebuffer(width, height, width * 4, Bitmap::ThirtyTwoBit, buffer.data());
            f.Rasterize(&framebuffer);
            
            return py::array_t<unsigned char>(
                {height, width, 4},
                {width * 4, 4, 1},
                buffer.data()
            );
        });

    // Global Helpers
    m.def("camera_count", []() {
        CameraList list;
        return list.Count();
    });

    m.def("get_camera", []() {
        return CameraManager::X().GetCamera();
    }, py::return_value_policy::reference);

    m.def("get_camera_by_index", [](int index) {
        CameraList list;
        if (index < 0 || index >= list.Count()) return (Camera*)nullptr;
        return CameraManager::X().GetCamera(list[index].UID());
    }, py::return_value_policy::reference);
}