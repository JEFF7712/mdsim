{
  description = "C++20 Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "cpp20";

          nativeBuildInputs = with pkgs; [
            gcc13
            cmake
            ninja
            gdb
          ];

          packages = with pkgs; [
            clang-tools
            cppcheck
            ovito
          ];

          shellHook = ''
            echo "C++20 Environment Loaded (GCC $(gcc --version | head -n1 | awk '{print $3}'))"
            export CC=gcc
            export CXX=g++
          '';
        };
      }
    );
}
