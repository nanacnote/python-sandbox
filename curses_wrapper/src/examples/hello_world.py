from time import sleep
from threading import active_count
from curses import A_REVERSE

from ..main import screen


if __name__ == "__main__":
    screen.add_ruler(116, 23).add_stage(1, 'stage_1_ref').add_title("New App")
    sleep(1)
    screen.add_str("first", 'data_1_ref').to_pos(5, 5).to_stage(1)
    sleep(1)
    screen.add_str("second", 'data_2_ref').to_pos(10, 10).to_stage(1)
    sleep(1)
    screen.update_str("new first", 'data_1_ref').with_attr(
        A_REVERSE).to_ruler(116, 23)
    sleep(1)
    print(active_count())
    # screen.update_status("data_1_ref change")
