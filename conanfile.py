from conan import ConanFile
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.cmake import CMakeDeps, CMakeToolchain

required_conan_version = ">=1.53.0"


class DevelopConan(ConanFile):
    name = "develop"
    version = "dev"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_opencv_gpu": [False, "True", "Thor"],
        "with_test": [True, False],
    }
    default_options = {
        "with_opencv_gpu": False,
        "with_test": False,
    }
    tool_requires = ["protobuf/3.21.12", "breakpad/cci.20210521"]

    @property
    def _min_cppstd(self):
        return 14

    def ffmpeg_confiure(self):
        self.options["ffmpeg"].avdevice = False
        self.options["ffmpeg"].with_asm = False
        self.options["ffmpeg"].with_libdav1d = False
        self.options["ffmpeg"].with_libaom = False
        self.options["ffmpeg"].with_libalsa = False
        self.options["ffmpeg"].with_pulse = False
        self.options["ffmpeg"].with_freetype = False
        self.options["ffmpeg"].with_libiconv = False
        self.options["ffmpeg"].with_openh264 = False
        self.options["ffmpeg"].with_vaapi = False
        self.options["ffmpeg"].with_vulkan = False
        self.options["ffmpeg"].with_xcb = False
        self.options["ffmpeg"].with_vdpau = False

    def configure(self):
        # specify glog options
        self.options["glog"].shared = True
        self.options["abseil"].shared = True
        # specifiy opencv options
        # self.options["opencv"].parallel = "openmp"
        self.options["opencv"].shared = True
        self.options["opencv"].with_ffmpeg = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_jpeg2000 = False
        self.options["opencv"].with_gtk = False
        self.options["opencv"].with_jpeg = "libjpeg-turbo"
        self.options["opencv"].text = False
        self.options["opencv"].with_wayland = False
        self.options["freeimage"].with_jpeg = "libjpeg-turbo"
        self.options["libyuv"].with_jpeg = "libjpeg-turbo"
        self.options["libtiff"].jpeg = "libjpeg-turbo"
        self.options["freeimage"].with_png = False
        self.options["freeimage"].with_tiff = False
        self.options["freeimage"].with_raw = False
        self.options["freeimage"].with_openexr = False
        self.options["acados"].shared = True
        if self.options.with_opencv_gpu != False:
            self.options["opencv"].with_cuda = True
            self.options["opencv"].cudaarithm = True
            self.options["opencv"].dnn = True
            self.options["opencv"].cudaimgproc = True
            self.options["opencv"].cudawarping = True
            self.options["opencv"].cuda_arch_bin = "7.2,7.5,8.6"

        self.options["ceres-solver"].use_glog = True
        self.options["pcl"].with_kdtree = True
        self.options["pcl"].with_filters = True
        self.options["pcl"].with_sample_consensus = True
        self.options["pcl"].with_search = True
        self.options["pcl"].with_segmentation = True
        self.options["pcl"].with_features = True
        self.options["pcl"].with_ml = True
        self.options["pcl"].with_geometry = True
        self.options["pcl"].with_2d = True
        self.options["proj"].with_tiff = False
        self.options["proj"].with_curl = False
        self.options["proj"].shared = True
        self.options["libyuv"].with_jpeg = False
        self.options["boost"].without_locale = True
        self.options["boost"].without_log = True
        self.options["boost"].without_test = True
        self.options["boost"].without_stacktrace = True
        self.options["boost"].without_fiber = True
        if self.settings.arch != "x86_64":
            self.ffmpeg_confiure()

    def requirements(self):
        self.requires("cppzmq/4.10.0")
        self.requires("protobuf/3.21.12")
        self.requires("benchmark/1.9.0")
        self.requires("libjpeg-turbo/3.0.0", force=True)
        self.requires("ceres-solver/2.1.0")
        self.requires("glog/0.6.0@transformer/stable", force=True)
        self.requires("abseil/20220623.1")
        self.requires("boost/1.75.0", force=True)
        self.requires("opencv/4.10.0")
        self.requires("libpng/1.6.40")
        self.requires("eigen/3.4.0")
        self.requires("toml11/3.7.0")
        self.requires("spdlog/1.9.2")
        self.requires("sqlite3/3.39.4", force=True)
        self.requires("yaml-cpp/0.8.0")
        self.requires("pcl/1.11.1")
        self.requires("libyuv/1880")
        self.requires("ald/0.1.11")  # for gnss driver
        self.requires("proj/9.5.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("nanoflann/1.4.3")
        self.requires("gtsam/4.1.1")
        self.requires("zstd/1.5.2")
        self.requires("libcurl/7.80.0")
        self.requires("acados/0.1.9")
        self.requires("adrss/1.1.0")
        self.requires("tcmap/1.0.3")
        self.requires("taskflow/3.8.0")
        self.requires("concurrentqueue/1.0.4")
        self.requires("cnpy/cci.20180601")

        if self.settings.arch == "x86_64":
            self.requires("langgemap/1.0.0")
            self.requires("gos-reinject/1.0.0")
        self.requires("zlib/1.2.13")
        self.requires("osqp/1.0.0-alpha")

        self.requires("xz_utils/5.4.5")
        self.requires("sml/1.1.11")
        self.requires("readerwriterqueue/1.0.6")
        self.requires("bshoshany-thread-pool/4.1.0")
        if self.options.with_test:
            self.requires("gtest/1.13.0@transformer/stable")
        self.requires("munkres/1.0.0")
        if self.options.with_innolidar:
            self.requires("inno_lidar/2.5.0")
        if self.options.with_rslidar:
            self.requires("rslidar_sdk/1.5.17")
        if self.options.with_hslidar:
            self.requires("hslidar_sdk/2.0.8")
        self.requires("quill/7.3.0")
        if self.settings.arch == "x86_64":
            self.requires("ffmpeg/cuda")
        else:  # armv8
            self.requires("ffmpeg/4.3.2")
        self.requires("asio/1.28.1", override=True)
        self.requires("flatbuffers/1.12.0")
        self.requires("jsoncpp/1.9.5")
        self.requires("mach_clock/0.3.1")
        self.requires("bshoshany-thread-pool/4.1.0")
        self.requires("argparse/3.1")  # 从3.2起不支持conan1
        self.requires("fmt/8.0.1")
        self.requires("breakpad/cci.20210521")
        self.requires("calib_result/1.0.3@e2e/dev")
        self.requires("libssh2/1.11.1")
        if self.settings.arch != "x86_64":
            self.requires("camera_monitor/0.0.6@")
        self.requires("libpcap/1.10.4")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"
        if self.options.with_test:
            tc.variables["BUILD_TESTING"] = True
        else:
            tc.variables["BUILD_TESTING"] = False
        tc.generate()
        tc = CMakeDeps(self)
        tc.generate()
        tc = VirtualRunEnv(self)
        tc.generate()
        tc = VirtualBuildEnv(self)
        tc.generate(scope="build")
