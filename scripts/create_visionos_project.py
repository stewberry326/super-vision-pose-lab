from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'visionos/TAPValidation.xcodeproj';P.mkdir(exist_ok=True)
ids=lambda n:f'B{n:023X}'
objects=[]
def add(n,body):objects.append(f'{ids(n)} = {{ {body} }};')
for i,file in enumerate(['TAPValidationApp.swift','ContentView.swift','TAPEngine.swift']):
 add(10+i,f'isa=PBXFileReference; lastKnownFileType=sourcecode.swift; path={file}; sourceTree="<group>";')
 add(20+i,f'isa=PBXBuildFile; fileRef={ids(10+i)};')
add(30,'isa=PBXFileReference; explicitFileType=wrapper.application; path=TAPValidation.app; sourceTree=BUILT_PRODUCTS_DIR;')
add(40,f'isa=PBXGroup; children=({ids(41)},{ids(42)}); sourceTree="<group>";')
add(41,f'isa=PBXGroup; path=TAPValidation; children=({ids(10)},{ids(11)},{ids(12)}); sourceTree="<group>";')
add(42,f'isa=PBXGroup; name=Products; children=({ids(30)}); sourceTree="<group>";')
add(50,f'isa=PBXSourcesBuildPhase; buildActionMask=2147483647; files=({ids(20)},{ids(21)},{ids(22)}); runOnlyForDeploymentPostprocessing=0;')
add(51,'isa=PBXFrameworksBuildPhase; buildActionMask=2147483647; files=(); runOnlyForDeploymentPostprocessing=0;')
add(52,'isa=PBXResourcesBuildPhase; buildActionMask=2147483647; files=(); runOnlyForDeploymentPostprocessing=0;')
script='set -e\\nmkdir -p "$BUILT_PRODUCTS_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"\\nfor unit in features query update; do\\n if [ -d "$SRCROOT/../coreml/$unit.mlpackage" ]; then\\n  xcrun coremlcompiler compile "$SRCROOT/../coreml/$unit.mlpackage" "$BUILT_PRODUCTS_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"\\n fi\\ndone'
# OpenStep literal escaping, keeping intentional \\n sequences.
script=script.replace('"','\\"')
add(53,f'isa=PBXShellScriptBuildPhase; buildActionMask=2147483647; files=(); inputPaths=(); outputPaths=(); runOnlyForDeploymentPostprocessing=0; shellPath=/bin/sh; shellScript="{script}"; name="Compile TAP models"; alwaysOutOfDate=1;')
add(60,f'isa=PBXNativeTarget; name=TAPValidation; productName=TAPValidation; productReference={ids(30)}; productType="com.apple.product-type.application"; buildPhases=({ids(50)},{ids(51)},{ids(52)},{ids(53)}); buildConfigurationList={ids(72)}; dependencies=(); buildRules=();')
add(61,f'isa=PBXProject; buildConfigurationList={ids(71)}; compatibilityVersion="Xcode 14.0"; developmentRegion=en; knownRegions=(en,Base); mainGroup={ids(40)}; productRefGroup={ids(42)}; projectDirPath=""; projectRoot=""; targets=({ids(60)}); attributes={{ LastUpgradeCheck=2700; }};')
base='SDKROOT=xros; XROS_DEPLOYMENT_TARGET=2.0; SWIFT_VERSION=5.0; CLANG_ENABLE_MODULES=YES;'
for num,name in [(80,'Debug'),(81,'Release')]:add(num,f'isa=XCBuildConfiguration; name={name}; buildSettings={{ {base} }};')
config='PRODUCT_BUNDLE_IDENTIFIER=dev.supervision.TAPValidation; PRODUCT_NAME="$(TARGET_NAME)"; GENERATE_INFOPLIST_FILE=YES; INFOPLIST_KEY_CFBundleDisplayName="TAP Validation"; INFOPLIST_KEY_UIApplicationSceneManifest_Generation=YES; TARGETED_DEVICE_FAMILY=7; SUPPORTED_PLATFORMS="xros xrsimulator"; CODE_SIGN_STYLE=Automatic; ENABLE_USER_SCRIPT_SANDBOXING=NO; SWIFT_STRICT_CONCURRENCY=minimal;'
for num,name in [(82,'Debug'),(83,'Release')]:add(num,f'isa=XCBuildConfiguration; name={name}; buildSettings={{ {base} {config} }};')
add(71,f'isa=XCConfigurationList; buildConfigurations=({ids(80)},{ids(81)}); defaultConfigurationIsVisible=0; defaultConfigurationName=Release;')
add(72,f'isa=XCConfigurationList; buildConfigurations=({ids(82)},{ids(83)}); defaultConfigurationIsVisible=0; defaultConfigurationName=Release;')
(P/'project.pbxproj').write_text('// !$*UTF8*$!\n{archiveVersion=1;classes={};objectVersion=56;objects={\n'+'\n'.join(objects)+f'\n}};rootObject={ids(61)};}}\n')
