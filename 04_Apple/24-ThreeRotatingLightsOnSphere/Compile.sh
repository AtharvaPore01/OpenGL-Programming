mkdir -p ThreeRotatingLightsOnSphere.app/Contents/MacOS

clang++ -o ThreeRotatingLightsOnSphere.app/Contents/MacOS/ThreeRotatingLightsOnSphere ThreeRotatingLightsOnSphere.mm -framework Cocoa -framework QuartzCore -framework OpenGL
