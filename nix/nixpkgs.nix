{ cudaSupport ? true
, overlays ? []
}:
import (import ./sources.nix).nixpkgs {
  config.allowUnfree = true;
  config.cudaSupport = true;
  overlays = (import ./overlays.nix { inherit cudaSupport; }) ++ overlays;
}
