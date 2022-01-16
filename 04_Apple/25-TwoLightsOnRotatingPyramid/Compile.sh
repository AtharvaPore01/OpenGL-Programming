mkdir -p TwoLightsOnRotatingPyramid.app/Contents/MacOS

clang++ -o TwoLightsOnRotatingPyramid.app/Contents/MacOS/TwoLightsOnRotatingPyramid TwoLightsOnRotatingPyramid.mm -framework Cocoa -framework QuartzCore -framework OpenGL
