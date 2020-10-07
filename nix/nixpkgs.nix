{ cudaSupport ? true
, overlays ? []
}:
import (import ./sources.nix).nixpkgs {
  config.allowUnfree = true;
  config.cudaSupport = cudaSupport;
  overlays = (import ./overlays.nix { inherit cudaSupport; }) ++ overlays;
}
