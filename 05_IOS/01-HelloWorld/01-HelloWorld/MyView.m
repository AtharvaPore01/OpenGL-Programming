//
//  MyView.m
//  01-HelloWorld
//
//  Created by user158739 on 1/28/20.
//

#import "MyView.h"

@implementation MyView
{
    NSString *centralText;
}

- (id)initWithFrame:(CGRect)frameRect
{
    self=[super initWithFrame:frameRect];
    if(self)
    {
        //initialise code here
        
        //set scene's background color
        [self setBackgroundColor:[UIColor whiteColor]];
        
        centralText=@"Hello World!!!";
        
        //GESTURE RECOGNITION
        //Tap Gesture Code
        UITapGestureRecognizer *singleTapGestureRecognizer=[[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer setNumberOfTouchesRequired:1];  //touch of 1 finger
        [singleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:singleTapGestureRecognizer];
        
        UITapGestureRecognizer *doubleTapGestureRecognizer=
        [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer setNumberOfTouchesRequired:1];
        [doubleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:doubleTapGestureRecognizer];
        
        //this allow to differentiate between single tap and double tap
        [singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
        
        //swipe gesture
        UISwipeGestureRecognizer *swipeGestureRecognizer=[[UISwipeGestureRecognizer alloc]initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer:swipeGestureRecognizer];
        
        //long press gesture
        UILongPressGestureRecognizer *longPressGestureRecognizer=[[UILongPressGestureRecognizer alloc]initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer:longPressGestureRecognizer];
    }
    return(self);
}

//only override draw rect:if we perform custom drawing.
//an empty implementation adversly affects performance during animation
- (void)drawRect:(CGRect)rect
{
    //black background
    UIColor *fillColor = [UIColor blackColor];
    [fillColor set];
    UIRectFill(rect);
    
    //dictionary with kvc
    NSDictionary *dictionaryForTextAttributes=[NSDictionary
        dictionaryWithObjectsAndKeys:[UIFont fontWithName:@"Helvetica" size:24],
                                               NSFontAttributeName,
                                               [UIColor greenColor],
                                               NSForegroundColorAttributeName,
                                               nil];
    
    CGSize textSize = [centralText sizeWithAttributes:dictionaryForTextAttributes];
    
    CGPoint point;
    point.x=(rect.size.width/2) - (textSize.width/2);
    point.y=(rect.size.height/2) - (textSize.height/2);
    
    [centralText drawAtPoint:point withAttributes:dictionaryForTextAttributes];
}

//to become first responder
- (BOOL)acceptsFirstResponder
{
    //code
    return (YES);
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    //code
}

-(void)onSingleTap:(UITapGestureRecognizer *)gr
{
    //code
    centralText = @"'onSingleTap' Event Occured";
    [self setNeedsDisplay];//repainting
}

-(void)onDoubleTap:(UITapGestureRecognizer *)gr
{
    //code
    centralText = @"'onDoubleTap' Event Occured";
    [self setNeedsDisplay];//repainting
}

-(void)onSwipe:(UISwipeGestureRecognizer *)gr
{
    //code
    [self release];
    exit(0);
}

-(void)onLongPress:(UILongPressGestureRecognizer *)gr
{
    //code
    centralText = @"'onLongPress' Event Occured";
    [self setNeedsDisplay];//repainting
}

- (void)dealloc
{
    //code
    [super dealloc];
}
@end
