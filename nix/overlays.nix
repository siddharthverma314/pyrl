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
      numpy = python-super.numpy.override { blas = super.mkl; };

      pytorch = python-super.pytorch.override {
        mklSupport = true;
        openMPISupport = true;
        cudaSupport = true;
        buildNamedTensor = true;
        cudaArchList = [
          "5.0"
          "5.2"
          "6.0"
          "6.1"
          "7.0"
          "7.5"
          "7.5+PTX"
        ];
      };

      opencv3 = python-super.opencv3.override {
        enableCuda = true;
        enableFfmpeg = true;
      };

      opencv4 = python-super.opencv4.override {
        enableCuda = true;
        enableFfmpeg = true;
      };

      mujoco-py = self.callPackage ./mujoco_py.nix {
        cudaSupport = false;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };

      mujoco-py_gpu = self.callPackage ./mujoco_py.nix {
        cudaSupport = true;
        mjKeyPath = /home/vsiddharth/secrets/mjkey.txt;
      };
    };

    python38 =
      super.python38.override { packageOverrides = self.pythonOverrides; };

    python3 = python38;
  })
]
