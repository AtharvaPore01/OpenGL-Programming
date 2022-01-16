mkdir -p GraphPaper.app/Contents/MacOS

clang++ -o GraphPaper.app/Contents/MacOS/GraphPaper GraphPaper.mm -framework Cocoa -framework QuartzCore -framework OpenGL
