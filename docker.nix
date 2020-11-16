{ pkgs, pkg, name, diskSize ? 1024 * 20, cudaSupport ? false }:
pkgs.dockerTools.buildImage {
  inherit name diskSize;
  contents = [pkgs.bashInteractive pkg];
  runAsRoot = if cudaSupport then '' 
    #!${pkgs.runtimeShell}
    mkdir -p /tmp
  '' else "";
  config = if cudaSupport then {
    Env = ["NVIDIA_DRIVER_CAPABILITIES=compute,utility" "NVIDIA_VISIBLE_DEVICES=all"];
  } else {};
}
