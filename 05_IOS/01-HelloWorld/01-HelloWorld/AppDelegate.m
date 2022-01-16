//
//  AppDelegate.m
//  01-HelloWorld
//
//  Created by user158739 on 1/28/20.
//

#import "AppDelegate.h"

#import "ViewController.h"

#import "MyView.h"

@implementation AppDelegate
{
@private
    UIWindow *mainWindow;
    ViewController *mainViewController;
    MyView *myView;
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
    myView=[[MyView alloc]initWithFrame:screenBounds];
    
    [mainViewController setView:myView];
    
    [myView release];
    
    //add the ViewController's view as subview to the window
    [mainWindow addSubview:[mainViewController view]];
    
    //make window key window and visible
    [mainWindow makeKeyAndVisible];
    
    return(YES);
}


- (void)applicationWillResignActive:(UIApplication *)application {
    // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
}


- (void)applicationDidEnterBackground:(UIApplication *)application {
    // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
}


- (void)applicationWillEnterForeground:(UIApplication *)application {
    // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
}


- (void)applicationDidBecomeActive:(UIApplication *)application {
    // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
}


- (void)applicationWillTerminate:(UIApplication *)application {
    // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
}

- (void)dealloc
{
    //code
    [myView release];
    
    [mainViewController release];
    
    [mainWindow release];
    
    [super dealloc];
}

@end
