from time import sleep
from curses import A_REVERSE

from ..main import dev, screen


dev(x_max=120, y_max=20)
if __name__ == "__main__":
    screen.add_stage(1, 'stage_1_ref').add_title("New App")
    sleep(1)
    screen.add_str("first", 'data_1_ref').to_pos(5, 5).to_stage(1)
    sleep(1)
    screen.add_str("second", 'data_2_ref').to_pos(10, 10).to_stage(1)
    sleep(1)
    screen.update_str("new first", 'data_1_ref').with_attr(
        A_REVERSE).to_ruler(60, 10)
    sleep(1)
    screen.update_status("data_1_ref change")
