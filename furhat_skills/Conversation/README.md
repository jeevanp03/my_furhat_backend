# Skill
Template skill
Main author: Nils Hagberg, Furhat Robotics

## Description
This is a skill template for a general interaction with Furhat. It's a suggestion and guide to create more advanced 
skills. The naming and structure of the files and packages is only a support and suggestion to the skill creator to 
create a useful structure - it has no influence on the skill.  

The intention of this skill template is to allow skill creators to focus on the "active" part of the interaction and 
get a pre-made handling of the start of the interaction, idling, users entering/leaving. Specific skills have specific 
requirements and should be designed accordingly - this is just an example of how the general scaffolding of a skill can 
be implemented.

This template will be subject to change at any time - and is not version handled or in any other way supported. 
If you have any suggestions, comments, or feedback we'd be happy to hear about them in the Slack Community.  

For more example skills go to: https://github.com/FurhatRobotics/

## Usage
Max number of users is set to: 5

No other specific requirements. 

## Flow
### Start of skill
Start in sleeping. Or start in idle. Idle is default. 

### Idle (IDLE)
Wait for users to enter - robot looks around randomly to keep itself busy. 

User enter - stop idling and go to WaitingForEngagedUser (IDLE)

No users for a while - goto Nap

### WaitingForEngagedUser (IDLE)
Try to catch their attention by seeking eye contact. A user that looks at the robot for a while will trigger the robot 
to go to an interaction state and start a conversation. 

Seek the attention of users standing closest, or random user. Alternate attention between people. 
Switch attention to new users entering, or users starting to attend the robot. 

No users present - go back to Idle. 

### Start (ACTIVE)
Maintain eye contact with user and wait for them to initiate the conversation with a greeting. 
Switch attention when another person speaks, or when current user looks away. 
Make a listening face and return their smiles

If they say hello, or how are you - respond politely but don't engage further, and find new user to attend. 
Glance at new users entering - but keep focus on the user speaking. 

No users attending  - go back to WaitingForEngagedUser
No users at all - go back to Idle

### Nap (SLEEP)
Sleep behavior. 
Wake up on user entering. 

After a longer while - goto DeepSleep

### DeepSleep (SLEEP)
Turn off projector
Only wake up from pressing wizard button. 