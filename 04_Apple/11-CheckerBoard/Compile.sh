mkdir -p CheckerBoard.app/Contents/MacOS

clang++ -o CheckerBoard.app/Contents/MacOS/CheckerBoard CheckerBoard.mm -framework Cocoa -framework QuartzCore -framework OpenGL
