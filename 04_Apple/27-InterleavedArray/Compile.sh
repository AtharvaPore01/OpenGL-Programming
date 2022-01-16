mkdir -p InterleavedArray.app/Contents/MacOS

clang++ -o InterleavedArray.app/Contents/MacOS/InterleavedArray InterleavedArray.mm -framework Cocoa -framework QuartzCore -framework OpenGL
