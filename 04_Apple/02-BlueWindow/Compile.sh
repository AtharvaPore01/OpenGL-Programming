mkdir -p BlueWindow.app/Contents/MacOS

clang++ -o BlueWindow.app/Contents/MacOS/BlueWindow BlueWindow.mm -framework Cocoa -framework QuartzCore -framework OpenGL
