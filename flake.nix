{
  description = "C++20 CUDA Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "cpp20-cuda-env";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            "/run/opengl-driver"
          ];

          nativeBuildInputs = with pkgs; [
            gcc13
            cmake
            ninja
            gdb
            cudatoolkit
            linuxPackages.nvidia_x11
          ];

          packages = with pkgs; [
            clang-tools
            cppcheck
            ovito
            python3
            python3Packages.pandas
            python3Packages.matplotlib
            python3Packages.scipy
          ];

          NIX_ENFORCE_NO_NATIVE = "0";

          shellHook = ''
            echo "C++20 CUDA Environment Loaded (GCC $(gcc --version | head -n1 | awk '{print $3}'))"
            export CC=gcc
            export CXX=g++
            export CUDA_PATH=${pkgs.cudatoolkit}
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
      }
    );
}