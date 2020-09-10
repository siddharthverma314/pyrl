[
  # add sources
  (self: super: {
    sources = import ./sources.nix;
  })

  # top-level pkgs overlays
  (self: super: {
    openmpi = super.openmpi.override { cudaSupport = true; };

    # batteries included :)
    ffmpeg = super.ffmpeg.override {
      nonfreeLicensing = true;
      nvenc = true; # nvidia support
    };
  })

  # python pkgs overlays
  (self: super: rec {
    packageOverrides = python-self: python-super: rec {
      blas = super.blas.override { blasProvider = super.mkl; };
      lapack = super.lapack.override { lapackProvider = super.mkl; };

      pytorch = python-super.pytorch.override {
        openMPISupport = true;
        cudaSupport = true;
      };

      opencv3 = python-super.opencv3.override {
        enableCuda = true;
        enableFfmpeg = true;
      };

      opencv4 = python-super.opencv4.override {
        enableCuda = true;
        enableFfmpeg = true;
      };

      mujoco-py_cpu = python-super.callPackage ./mujoco_py.nix {
        mesa = super.mesa;
        cudaSupport = false;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };

      mujoco-py_gpu = python-super.callPackage ./mujoco_py.nix {
        mesa = super.mesa;
        cudaSupport = true;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };

      mujoco-py = mujoco-py_gpu;

      cpprb = python-super.callPackage ./cpprb.nix {};

      gym = python-super.gym.overrideAttrs (old: {
        postPatch = ''
          substituteInPlace setup.py \
            --replace "pyglet>=1.2.0,<=1.3.2" "pyglet" \
            --replace "cloudpickle>=1.2.0,<1.4.0" "cloudpickle" \
        '';
      });

      # self-made packages
      glfw = python-super.callPackage ./glfw.nix {};
      flatten-dict = python-super.callPackage ./flatten-dict.nix {};
      scikit-video = python-super.callPackage ./scikit-video.nix {};
    };

    python38 = super.python38.override { inherit packageOverrides; };
    python3 = python38;
  })
]
