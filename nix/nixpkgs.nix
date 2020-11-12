{ cudaSupport ? true
, overlays ? []
, python ? "python38"
, inputs
}:
import inputs.nixpkgs {
  config.allowUnfree = true;
  config.cudaSupport = cudaSupport;
  overlays = (import ./overlays.nix { inherit cudaSupport python inputs; }) ++ overlays;
}
