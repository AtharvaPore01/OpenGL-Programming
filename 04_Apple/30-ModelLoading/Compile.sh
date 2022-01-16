mkdir -p ModelLoading.app/Contents/MacOS

clang++ -o ModelLoading.app/Contents/MacOS/ModelLoading ModelLoading.mm -framework Cocoa -framework QuartzCore -framework OpenGL
