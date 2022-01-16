mkdir -p DeathlyHallows.app/Contents/MacOS

clang++ -o DeathlyHallows.app/Contents/MacOS/DeathlyHallows DeathlyHallows.mm -framework Cocoa -framework QuartzCore -framework OpenGL
