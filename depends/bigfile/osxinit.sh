curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall > ~/uninstall_homebrew
chmod +x ~/uninstall_homebrew
~/uninstall_homebrew -fq

cat > conda_build_config.yaml <<END
CONDA_BUILD_SYSROOT:
- $(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk # [osx]
END
ls -al $(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs
cat conda_build_config.yaml
