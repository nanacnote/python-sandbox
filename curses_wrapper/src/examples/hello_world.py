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
    screen.update_entity([(1, "new first")], 'data_1_ref').with_attr(
        A_REVERSE)
    sleep(1)
    screen.update_status("data_1_ref change")
    sleep(1)
    screen.add_button("button", 'btn_ref_1').to_pos(0, 0).to_stage(1)
