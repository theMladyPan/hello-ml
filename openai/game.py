import arcade
 
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500
 
# Creates the game window
arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game")
 
# Sets background color
arcade.set_background_color(arcade.color.WHEAT)
 
# Clears the screen
arcade.start_render()
  
# Draws our square
arcade.draw_rectangle_filled(200, 250, 25, 25, arcade.color.BLUE)

# Sets the player's movement speed
MOVEMENT_SPEED = 3
  
# Variables that keep track of the player's position
player_x = 200
player_y = 250
 
# Defines the player's movement
def on_update(delta_time):
    global player_x, player_y
 
    if arcade.check_for_collision(player_x, player_y, 10, 10,200,250,25,25):
        print("Collision")
    else:
        if arcade.key.UP:
            player_y += MOVEMENT_SPEED
        if arcade.key.DOWN:
            player_y -= MOVEMENT_SPEED
        if arcade.key.LEFT:
            player_x -= MOVEMENT_SPEED
        if arcade.key.RIGHT:
            player_x += MOVEMENT_SPEED
  
    # Draws the player
    arcade.draw_rectangle_filled(player_x, player_y, 10, 10,
                                arcade.color.YELLOW)
  
# Clears the screen
arcade.finish_render()
  
# Keeps the window open until the user hits the 'close' button
arcade.run()
