#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include "cameralibrary.h"

namespace py = pybind11;
using namespace CameraLibrary;

PYBIND11_MODULE(optitrack_cam, m) {
    m.doc() = "MAVIS OptiTrack Camera Library bindings";
    m.attr("BINDING_API_VERSION") = 2;
    m.attr("CAMERA_RECORDING_API_VERSION") = 1;
    m.attr("OPTIHUB_SYNC_CONTROL_API_VERSION") = 1;

    // OptiHub 2 exposes synchronization configuration. Frame lock telemetry
    // is documented for eSync only; never report it as OptiHub input activity.
    m.def("get_sync_status", []() {
        sSyncSettings settings;
        CameraManager::X().GetSyncSettings(settings);
        HubList hubs;
        py::dict result;
        result["available"] = hubs.Count() > 0;
        result["hub_count"] = hubs.Count();
        const char* name = CameraManager::X().SyncDeviceName();
        result["device_name"] = name ? name : "";
        result["sync_mode"] = (int)settings.Mode;
        result["sync_type"] = (int)settings.SyncType;
        result["camera_sync"] = (int)settings.CameraSync;
        result["input_source"] = (int)settings.SyncInputSource;
        result["input_trigger"] = (int)settings.SyncInputTrigger;
        result["input_divider"] = settings.SyncInputDivider;
        result["global_offset_us"] = settings.GlobalSyncOffset;
        result["internal_frequency_hz"] = settings.InternalSyncGeneratorFrequency;
        result["output1_type"] = (int)settings.SyncOutput1PulseDuration;
        result["output1_polarity"] = (int)settings.SyncOutput1Polarity;
        result["input_edge_recording_supported"] = false;
        result["reason"] = "OptiHub 2 SDK exposes configuration, not input levels or timestamped edges";
        return result;
    });

    m.def("configure_sync_preset", [](const std::string& preset) {
        const bool external = preset == "vicon_external_50hz";
        if (!external && preset != "internal_50hz")
            throw py::value_error("Unknown OptiHub sync preset");
        HubList hubs;
        auto& manager = CameraManager::X();
        const auto features = manager.GetSyncFeatures();
        if (hubs.Count() == 0 || !features.CustomSyncSupport)
            throw std::runtime_error("No custom-sync-capable OptiHub is connected");
        sSyncSettings previous;
        manager.GetSyncSettings(previous);
        sSyncSettings settings = previous;
        settings.Mode = SyncModeCustom;
        // Preserve the existing OptiSync/WiredSync camera topology.
        settings.SyncInputSource = external ? SyncInputSourceSyncExternal : SyncInputSourceInternal;
        settings.SyncInputTrigger = SyncInputSourceSyncExternalRisingEdge;
        settings.SyncInputDivider = 1;
        if (features.SyncInputMultiplier)
            settings.SyncInputMultiplier = 1;
        settings.GlobalSyncOffset = 0;
        if (!external)
            settings.InternalSyncGeneratorFrequency = 50.0f;
        try {
            manager.ApplySyncSettings(settings);
            sSyncSettings actual;
            manager.GetSyncSettings(actual);
            if (actual.Mode != settings.Mode || actual.SyncType != previous.SyncType ||
                actual.SyncInputSource != settings.SyncInputSource ||
                (external && actual.SyncInputTrigger != settings.SyncInputTrigger) ||
                actual.SyncInputDivider != 1 ||
                (features.SyncInputMultiplier && actual.SyncInputMultiplier != 1) ||
                actual.GlobalSyncOffset != 0 ||
                (!external && std::fabs(actual.InternalSyncGeneratorFrequency - 50.0f) > .01f))
                throw std::runtime_error("SDK sync settings did not match the requested preset");
        } catch (...) {
            manager.ApplySyncSettings(previous);
            throw;
        }
        // External frequency is supplied by Vicon, not generated or measured here.
    }, py::arg("preset"));

    m.def("configure_exposure_output", []() {
        HubList hubs;
        auto& manager = CameraManager::X();
        const auto features = manager.GetSyncFeatures();
        if (hubs.Count() == 0 || !features.CustomSyncSupport || features.SyncOutputCount < 1)
            throw std::runtime_error("No configurable OptiHub output is available");
        sSyncSettings previous;
        manager.GetSyncSettings(previous);
        sSyncSettings settings = previous;
        settings.Mode = SyncModeCustom;
        settings.SyncOutput1PulseDuration = SyncOutputPulseDurationExposureTime;
        settings.SyncOutput1Phase = SyncOutputInPhase;
        settings.SyncOutput1Polarity = SyncOutputPolarityNormal;
        try {
            manager.ApplySyncSettings(settings);
            sSyncSettings actual;
            manager.GetSyncSettings(actual);
            if (actual.Mode != settings.Mode || actual.SyncOutput1PulseDuration != settings.SyncOutput1PulseDuration ||
                actual.SyncOutput1Phase != settings.SyncOutput1Phase ||
                actual.SyncOutput1Polarity != settings.SyncOutput1Polarity ||
                actual.SyncInputSource != previous.SyncInputSource || actual.SyncType != previous.SyncType)
                throw std::runtime_error("SDK did not apply the exposure output configuration");
        } catch (...) {
            manager.ApplySyncSettings(previous);
            throw;
        }
    });

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
        .def("capture_info", [](Camera &c) {
            py::dict result;
            result["serial"] = c.Serial();
            result["model"] = c.Model();
            result["frame_rate_setting"] = c.FrameRate();
            const char* name = c.Name();
            result["name"] = name ? name : "";
            return result;
        })
        .def("start", &Camera::Start)
        .def("stop", &Camera::Stop, py::arg("turn_numeric_off") = true)
        .def("width", &Camera::Width)
        .def("height", &Camera::Height)
        .def("set_video_type", [](Camera &c, int type) { c.SetVideoType((Core::eVideoMode)type); })
        .def("set_aec", &Camera::SetAEC)
        .def("set_agc", &Camera::SetAGC)
        .def("set_exposure", &Camera::SetExposure)
        .def("get_exposure", &Camera::Exposure)
        .def("get_intensity", &Camera::Intensity)
        .def("min_intensity", &Camera::MinimumIntensity)
        .def("max_intensity", &Camera::MaximumIntensity)
        .def("set_intensity", [](Camera &c, int value) {
            if (value < c.MinimumIntensity() || value > c.MaximumIntensity())
                throw py::value_error("IR intensity is outside the camera's reported limits");
            c.SetIntensity(value);
            return c.Intensity();
        })
        .def("is_high_power_supported", &Camera::IsHighPowerModeSupported)
        .def("is_high_power_available", &Camera::IsHighPowerModeAvailable)
        .def("get_high_power", &Camera::HighPowerMode)
        .def("set_high_power", [](Camera &c, bool enabled) {
            if (!c.IsHighPowerModeSupported())
                throw py::value_error("High power is not adjustable on this camera");
            if (enabled && !c.IsHighPowerModeAvailable())
                throw py::value_error("High power is unavailable with the current camera/hub connection");
            c.SetHighPowerMode(enabled);
            return c.HighPowerMode();
        })
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
        .def("timing", [](Frame &f) {
            py::dict result;
            result["sdk_frame_id"] = f.FrameID();
            result["sdk_timestamp"] = f.TimeStamp();
            result["hardware_timestamp_valid"] = f.IsHardwareTimeStamp();
            if (f.IsHardwareTimeStamp()) {
                result["hardware_timestamp_ticks"] = f.HardwareTimeStamp();
                result["hardware_timestamp_frequency"] = f.HardwareTimeFreq();
            }
            return result;
        })
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
