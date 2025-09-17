from conan import ConanFile
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.cmake import CMakeDeps, CMakeToolchain

required_conan_version = ">=1.53.0"


class DevelopConan(ConanFile):
    name = "develop"
    version = "dev"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    options = {
        "with_test": [True, False],
    }
    default_options = {
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
        self.options["freeimage"].with_jpeg = "libjpeg-turbo"
        self.options["libyuv"].with_jpeg = "libjpeg-turbo"
        self.options["libtiff"].jpeg = "libjpeg-turbo"
        self.options["freeimage"].with_png = False
        self.options["freeimage"].with_tiff = False
        self.options["freeimage"].with_raw = False
        self.options["freeimage"].with_openexr = False
        self.options["acados"].shared = True

        self.options["ceres-solver"].use_glog = True
        self.options["proj"].with_tiff = False
        self.options["proj"].with_curl = False
        self.options["proj"].shared = True
        self.options["libyuv"].with_jpeg = False
        if self.settings.arch != "x86_64":
            self.ffmpeg_confiure()

    def requirements(self):
        self.requires("cppzmq/4.10.0")
        self.requires("protobuf/3.21.12")
        self.requires("benchmark/1.9.0")
        self.requires("libjpeg-turbo/3.0.0", force=True)
        self.requires("ceres-solver/2.1.0")
        self.requires("glog/0.7.1", force=True)
        self.requires("abseil/20230125.2")
        self.requires("libpng/1.6.40")
        self.requires("eigen/3.4.0")
        self.requires("toml11/3.7.0")
        self.requires("sqlite3/3.39.4", force=True)
        self.requires("yaml-cpp/0.8.0")
        self.requires("pcl/1.13.1")
        self.requires("libyuv/1880")
        self.requires("proj/9.5.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("nanoflann/1.4.3")
        self.requires("zstd/1.5.2")
        self.requires("libcurl/7.80.0")
        self.requires("taskflow/3.8.0")
        self.requires("concurrentqueue/1.0.4")
        self.requires("cnpy/cci.20180601")
        self.requires("numcpp/2.12.1")
        self.requires("zlib/1.2.13")
        self.requires("osqp/0.6.3")

        self.requires("xz_utils/5.4.5")
        self.requires("sml/1.1.11")
        self.requires("readerwriterqueue/1.0.6")
        self.requires("bshoshany-thread-pool/4.1.0")
        self.requires("gtest/1.15.0")
        self.requires("quill/7.3.0")
        self.requires("ffmpeg/4.3.2")
        self.requires("asio/1.28.1", override=True)
        self.requires("flatbuffers/1.12.0")
        self.requires("jsoncpp/1.9.5")
        self.requires("bshoshany-thread-pool/4.1.0")
        self.requires("argparse/3.1")  # 从3.2起不支持conan1
        self.requires("fmt/8.0.1")
        self.requires("libssh2/1.11.1")
        self.requires("libpcap/1.10.4")