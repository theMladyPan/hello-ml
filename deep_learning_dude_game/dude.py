import arcade

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
SCREEN_TITLE = "Minimal Racing Game"

_G = 9.81

class Game(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.BLACK)

        # Set up the player car 
        self.dude_sprite = arcade.Sprite("./graphics/dude.png")
        self.character_sprite_list = arcade.load_spritesheet(
            "graphics/characters.png", 
            32, 
            32, 
            23,
            23,
            
        )
        self.dude_sprite.texture = self.character_sprite_list[0]
        # resize the sprite
        self.dude_sprite.scale = 2
        self.dude_sprite.center_x = 50
        self.dude_sprite.center_y = 50
        
        self.dude_speed_up = 0
        self.dude_move_right = 0
        
        self._g = _G
        self.frame_ctr = 0
        
        # Set up the scrolling background
        self.background_sprite = arcade.Sprite("./graphics/city_background_night.png")
        self.background_sprite.scale = 0.5
        self.background_sprite.speed = 1
        self.background_sprite.center_x = SCREEN_WIDTH / 2
        self.background_sprite.center_y = SCREEN_HEIGHT / 2
        
        self.frame_index = 0

    def on_draw(self):
        arcade.start_render()
        
        # Draw the scrolling background
        self.background_sprite.draw()

        # Draw the player car
        self.dude_sprite.draw()

    def update(self, delta_time):
        # Update the player character animation frame
        if self.dude_speed_up == 0:
            self.frame_ctr += 1
            if self.frame_ctr % 5 == 0:
                self.frame_index += 1
            if self.frame_index >= 4:
                self.frame_index = 0
            self.dude_sprite.texture = self.character_sprite_list[self.frame_index]
        else:
            self.dude_sprite.texture = self.character_sprite_list[4]

        if self.dude_sprite.center_y < 50:
            self.dude_speed_up = 0
            self.dude_sprite.center_y = 50
        else:
            self.dude_speed_up -= self._g * delta_time
            
        self.dude_sprite.center_y += self.dude_speed_up
        
        self.background_sprite.speed += 0.002
        self._g += 0.002
            
        # Update the scrolling background
        self.background_sprite.center_x -= self.background_sprite.speed
        if self.background_sprite.right <= SCREEN_WIDTH:
            self.background_sprite.center_x = self.background_sprite.width / 2


    def on_key_press(self, key, modifiers):
        """if key == arcade.key.LEFT:
            self.dude_move_right = -1
        elif key == arcade.key.RIGHT:
            self.dude_move_right = 1"""
        if key in [arcade.key.UP, arcade.key.SPACE]:
            # self.dude_move_up = 1
            self.dude_speed_up = self._g / 2
        # elif key == arcade.key.DOWN:
        #     self.dude_move_up = -1

    def on_key_release(self, key, modifiers): ...
        # if key == arcade.key.LEFT or key == arcade.key.RIGHT:
        #     self.dude_move_right = 0
        # elif key == arcade.key.UP or key == arcade.key.DOWN:
        #     self.dude_move_up = 0

if __name__ == "__main__":
    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()
