[
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

    pythonOverrides = python-self: python-super: {
      #numpy = python-super.numpy.override { blas = super.mkl; };

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

      mujoco-py = python-self.callPackage ./mujoco_py.nix {
        mesa = super.mesa;
        cudaSupport = false;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };

      mujoco-py_gpu = python-self.callPackage ./mujoco_py.nix {
        mesa = super.mesa;
        cudaSupport = true;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };

      cpprb = python-self.callPackage ./cpprb.nix {};

      gym = python-super.gym.overrideAttrs (old: {
        postPatch = ''
          substituteInPlace setup.py \
            --replace "pyglet>=1.2.0,<=1.3.2" "pyglet" \
            --replace "cloudpickle>=1.2.0,<1.4.0" "cloudpickle" \
        '';
      });

      glfw = python-self.callPackage ./glfw.nix {};

      flatten-dict = python-self.callPackage ./flatten-dict.nix {};
    };

    python38 =
      super.python38.override { packageOverrides = self.pythonOverrides; };

    python3 = python38;
  })
]
