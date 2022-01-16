mkdir -p RoboticArm.app/Contents/MacOS

clang++ -o RoboticArm.app/Contents/MacOS/RoboticArm RoboticArm.mm -framework Cocoa -framework QuartzCore -framework OpenGL
