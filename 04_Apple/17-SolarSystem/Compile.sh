mkdir -p SolarSystem.app/Contents/MacOS

clang++ -o SolarSystem.app/Contents/MacOS/SolarSystem SolarSystem.mm -framework Cocoa -framework QuartzCore -framework OpenGL
