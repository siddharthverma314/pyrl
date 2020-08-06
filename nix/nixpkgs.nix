let
  pkgs = import <nixpkgs> {};
  nixpkgs-src = pkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "181179c53b7969986fd5067bba6f03fdeaef7fd4";
    sha256 = "1h715za7ma4swaxswi7q48yq46l63znczj9mg6nhig83zpjfn18k";
  };
in
import nixpkgs-src {
  config.allowUnfree = true;
  config.cudaSupport = true;
  overlays = import ./overlays.nix;
}
