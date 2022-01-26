import curses

from typing import Any
from threading import Thread
from signal import signal, SIGWINCH
from uuid import uuid4

from enum import Enum, unique


@unique
class StageProps(Enum):
    NAME=1


@unique
class EntityProps(Enum):
    PAYLOAD=1
    STAGE=2
    X_CORD=3
    Y_CORD=4
    ATTRIBUTES=5
    TYPE=6

class _dict(dict):
    def __setitem__(self, k, v):
        CursesUI.tick = True
        super(_dict, self).__setitem__(k, v)

    def update(self, d):
        CursesUI.tick = True
        super(_dict, self).update(d)


# TODO:
# implement on_screen method for redraw event
# implement on_tick method for redraw event
# implement optional entity and stage referrence
# trap kill signall
# on termination reset curs_set

class CursesUI:
    scr: Any = None
    frame: Any = None
    tick: Any = None
    sig: Any = None
    dev_mode: Any = None
    vis_stage = 1

    @staticmethod
    def stop(sig, *args):
        CursesUI.sig = sig
        curses.endwin()
        print(args)



    def __init__(self):
        self.__ruler_x_min, self.__ruler_x_max = 100, 100
        self.__ruler_y_min, self.__ruler_y_max = 10, 10

        self.__status_uid = uuid4()

        self.__cur_stage = tuple()
        self.__cur_entity = tuple()

        self.__refs_state = _dict()
        self.__stages_state = _dict()
        self.__entities_state = _dict()

        self.__loop = Thread(target=self.__init_loop)
        self.__loop.start()

    def __init_loop(self):
        def main(stdscr):
            curses.curs_set(0)

            stdscr.nodelay(True)
            stdscr.clear()
            stdscr.getch()

            CursesUI.frame = curses.newwin(curses.LINES, curses.COLS, 0, 0)
            CursesUI.frame.border()

            CursesUI.content = curses.newwin(
                curses.LINES - 6, curses.COLS - 6, 4, 4)
            CursesUI.content.overwrite(CursesUI.frame)

        curses.wrapper(main)
        self.__init_status()
        self.__ruler_x_max = ((curses.LINES-4)//2)+7
        self.__ruler_y_max = ((curses.COLS - 4)//4)+98

        if CursesUI.dev_mode:
            self.__draw_grid()

        while True:
            if CursesUI.sig == SIGWINCH:
                CursesUI.sig = None
                self.__loop.join()
                break

            if CursesUI.tick:
                self.__draw_content()
                self.__draw_frame()
                curses.doupdate()
                CursesUI.tick = False


    def __init_status(self):
        """
        Adds initial status text at bottom of frame window
        """
        self.__create_entity(self.__status_uid)
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.X_CORD.value] = 4
        props[EntityProps.Y_CORD.value] = curses.LINES-3
        props[EntityProps.STAGE.value] = 0
        self.__entities_state.update({entity[0]: tuple(props)})
        return self


    def _init_dev_mode(self, **kwargs):
        """
        Adds ruler markings to screen
        """
        self.__ruler_x_max = kwargs.get("x_max")
        self.__ruler_y_max = kwargs.get("y_max")


    def __draw_grid(self):
        """
        Draws a grid layout onto the screen.
        """
        for i in range(2, (curses.LINES - 4)//2):
            CursesUI.frame.addstr((i * 2), 1, str(i + 8))
            CursesUI.frame.hline((i * 2), 3, curses.ACS_HLINE, 1)

        for i in range(1, (curses.COLS - 4)//4):
            CursesUI.frame.addstr(1, (i * 4), str(i + 99))
            CursesUI.frame.vline(2, ((i * 4) + 1), curses.ACS_VLINE, 1)

    def __draw_frame(self):
        """
        Adds a header and status bar
        """
        entites = set(self.__entities_state.values())
        for entity in entites:
            if not entity[EntityProps.STAGE.value]:
                CursesUI.frame.addstr(
                    entity[EntityProps.Y_CORD.value],
                    entity[EntityProps.X_CORD.value],
                    entity[EntityProps.PAYLOAD.value],
                    entity[EntityProps.ATTRIBUTES.value]
                )
                CursesUI.frame.noutrefresh()

    def __draw_content(self):
        """
        Adds all content to screen
        """
        entites = set(self.__entities_state.values())
        for entity in entites:
            if entity[EntityProps.STAGE.value] == self.vis_stage:
                if entity[EntityProps.TYPE.value] == 1:
                    pass
                elif entity[EntityProps.TYPE.value] == 2:
                    pass
                else:
                    CursesUI.content.addstr(
                        entity[EntityProps.Y_CORD.value],
                        entity[EntityProps.X_CORD.value],
                        entity[EntityProps.PAYLOAD.value],
                        entity[EntityProps.ATTRIBUTES.value]
                    )
                CursesUI.content.noutrefresh()

    def __create_stage(self, ref):
        """
        Initialises a stage with base attributes
        """
        uid = uuid4()
        self.__stages_state.update({uid: (uid, 1)})
        self.__cur_stage = self.__stages_state[uid]

        if ref:
            self.__refs_state.update({ref: uid})

    def __create_entity(self, ref):
        """
        Initialises an entityents with base attributes
        """
        uid = uuid4()
        self.__entities_state.update({uid: (uid, "...", 1, 0, 0, 0, 0)})
        self.__cur_entity = self.__entities_state[uid]

        if ref:
            self.__refs_state.update({ref: uid})

    def add_stage(self, pl, ref=None):
        self.__create_stage(ref)
        stage = self.__cur_stage
        props = list(stage)
        props[StageProps.NAME.value] = pl
        self.__stages_state.update({stage[0]: tuple(props)})
        return self

    def add_title(self, pl, ref=None):
        self.__create_entity(ref)
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.PAYLOAD.value] = f' {pl} '
        props[EntityProps.X_CORD.value] = 4
        props[EntityProps.Y_CORD.value] = 0
        props[EntityProps.STAGE.value] = 0
        self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def add_str(self, pl, ref=None):
        self.__create_entity(ref)
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.PAYLOAD.value] = pl
        self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def add_button(self, pl, ref):
        self.__create_entity(ref)
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.PAYLOAD.value] = pl
        props[EntityProps.TYPE.value] = 1
        self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def add_input(self, pl, ref):
        self.__create_entity(ref)
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.PAYLOAD.value] = pl
        props[EntityProps.TYPE.value] = 2
        self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def to_pos(self, x, y):
        entity = self.__entities_state.get(self.__cur_entity[0])
        if entity and entity[EntityProps.STAGE.value]:
            props = list(entity)
            props[EntityProps.X_CORD.value] = x
            props[EntityProps.Y_CORD.value] = y
            self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def to_ruler(self, x, y):
        entity = self.__entities_state.get(self.__cur_entity[0])
        if entity and entity[EntityProps.STAGE.value]:
            props = list(entity)
            props[EntityProps.X_CORD.value] = (((x-self.__ruler_x_min)*100)//(self.__ruler_x_max-self.__ruler_x_min)*(curses.COLS-10))//100
            props[EntityProps.Y_CORD.value] = (((y-self.__ruler_y_min)*100)//(self.__ruler_y_max-self.__ruler_y_min)*(curses.LINES-10))//100
            self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def to_stage(self, num):
        entity = self.__entities_state.get(self.__cur_entity[0])
        if entity and entity[EntityProps.STAGE.value]:
            props = list(entity)
            props[EntityProps.STAGE.value] = num
            self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def with_attr(self, attr):
        entity = self.__entities_state.get(self.__cur_entity[0])
        if entity and entity[EntityProps.STAGE.value]:
            props = list(entity)
            props[EntityProps.ATTRIBUTES.value] = attr
            self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def update_status(self, pl):
        self.__cur_entity = self.__entities_state[self.__refs_state[self.__status_uid]]
        entity = self.__cur_entity
        props = list(entity)
        props[EntityProps.PAYLOAD.value] = pl
        self.__entities_state.update({entity[0]: tuple(props)})
        return self

    def update_entity(self, pl, ref):
        self.__cur_entity = self.__entities_state[self.__refs_state[ref]]
        entity = self.__cur_entity
        props = list(entity)
        for tup in pl:
            props[tup[0]] = tup[1]
        self.__entities_state.update({entity[0]: tuple(props)})
        return self




screen = CursesUI()
signal(SIGWINCH, screen.stop)


def dev(**kwargs):
    CursesUI.dev_mode = True
    screen._init_dev_mode(**kwargs)

