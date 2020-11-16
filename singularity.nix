{ pkgs, pkg, name, diskSize ? 1024 * 20 }:
pkgs.singularity-tools.buildImage {
  inherit name diskSize;
  contents = [ pkg ];
}
