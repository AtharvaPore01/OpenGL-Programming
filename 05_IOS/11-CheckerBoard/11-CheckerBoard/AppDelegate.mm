#import "AppDelegate.h"

#import "ViewController.h"

#import "GLESView.h"

@implementation AppDelegate
{
@private
    UIWindow *mainWindow;
    ViewController *mainViewController;
    GLESView *glesView;
}


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    //code
    
    //get screen bounds for fullscreen
    CGRect screenBounds=[[UIScreen mainScreen]bounds];
    
    //initialise window variable corresponding to screen bounds
    mainWindow=[[UIWindow alloc]initWithFrame:screenBounds];
    
    mainViewController=[[ViewController alloc]init];
    
    [mainWindow setRootViewController:mainViewController];
    
    //initialise view variable corresponding to screen
    glesView=[[GLESView alloc]initWithFrame:screenBounds];
    
    [mainViewController setView:glesView];
    
    [glesView release];
    
    //add the ViewController's view as subview to the window
    [mainWindow addSubview:[mainViewController view]];
    
    //make window key window and visible
    [mainWindow makeKeyAndVisible];
    
    //start animation
    [glesView startAnimation];
    
    return(YES);
}


- (void)applicationWillResignActive:(UIApplication *)application
{
    //code
    [glesView stopAnimation];
}


- (void)applicationDidEnterBackground:(UIApplication *)application {
    // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
}


- (void)applicationWillEnterForeground:(UIApplication *)application {
    // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
}


- (void)applicationDidBecomeActive:(UIApplication *)application
{
    //code
    [glesView startAnimation];
}


- (void)applicationWillTerminate:(UIApplication *)application
{
    //code
    [glesView stopAnimation];
}

- (void)dealloc
{
    //code
    [glesView release];
    
    [mainViewController release];
    
    [mainWindow release];
    
    [super dealloc];
}

@end
