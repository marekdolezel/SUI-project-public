import hexutil
from json.decoder import JSONDecodeError
import logging
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt5.QtGui import QPainter, QColor, QPolygon, QPen, QFont
from PyQt5.QtCore import QPoint, Qt, QRectF, QTimer


# dirty hack to limit the number of transfers per turn even for humans
# The limit needs to be set by user of the module (client script)
MAX_TRANSFERS_PER_TURN = None
# The running counter is handled by MainWindow.mousePressEvent (increments) and ClientUI.handle_end_turn_button (clearing)
nb_transfers_this_turn = 0


def player_color(player_name):
    """Return color of a player given his name
    """
    return {
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
        4: (255, 255, 0),
        5: (0, 255, 255),
        6: (255, 0, 255),
        7: (224, 224, 224),
        8: (153, 153, 255)
    }[player_name]


class MainWindow(QWidget):
    """Main window of the GUI containing the game board
    """
    def __init__(self, game, area_text_fn=lambda area: str(area.get_dice())):
        """
        Parameters
        ----------
        game : Game
        """
        self.logger = logging.getLogger('GUI')
        super(MainWindow, self).__init__()
        self.qp = QPainter()

        self.game = game
        self.board = game.board
        self.areas_mapping = {}
        for i, area in self.board.areas.items():
            for h in area.get_hexes():
                self.areas_mapping[h] = i

        self.font = QFont('Helvetica', 16)
        self.pen = QPen()
        self.pen.setWidth(2)

        self.activated_area_name = None
        self.area_text_fn = area_text_fn

    def paintEvent(self, event):
        self.qp.begin(self)
        self.draw_areas()
        self.qp.end()

    def set_area_text_fn(self, area_text_fn):
        self.area_text_fn = area_text_fn

    def draw_areas(self):
        """Draw areas in the game board
        """
        if self.game.draw_battle:
            self.game.draw_battle = False
        size = self.size()
        x = size.width()
        y = size.height()

        hexgrid = hexutil.HexGrid(10)

        self.qp.setPen(Qt.NoPen)
        self.qp.translate(x // 2, y // 2)

        for k, area in self.board.areas.items():
            lines = []
            first_hex = True

            color = player_color(area.get_owner_name())
            if self.activated_area_name == int(k):
                color = (170 + color[0] // 3, 170 + color[1] // 3, 170 + color[2] // 3)
            self.qp.setBrush(QColor(*color))
            self.qp.setPen(Qt.NoPen)
            for h in area.get_hexes():
                polygon = QPolygon([QPoint(*corner) for corner in hexgrid.corners(h)])
                self.qp.drawPolygon(polygon)

                if first_hex:
                    self.qp.save()
                    rect = QRectF(*hexgrid.bounding_box(h))
                    self.qp.setBrush(QColor(0, 0, 0))
                    self.qp.setPen(self.pen)
                    self.qp.setFont(self.font)
                    self.qp.setRenderHint(QPainter.TextAntialiasing)

                    self.qp.drawText(rect, Qt.AlignCenter, self.area_text_fn(area))
                    first_hex = False
                    self.qp.restore()

                for n in h.neighbours():
                    if n not in area.get_hexes():
                        line = []
                        for corner in hexgrid.corners(h):
                            if corner in hexgrid.corners(n):
                                line.append(corner)
                        lines.append(line)

            self.qp.save()
            pen = QPen()
            pen.setWidth(3)
            self.qp.setPen(pen)
            self.qp.setBrush(QColor())
            self.qp.setRenderHint(QPainter.Antialiasing)
            for line in lines:
                self.qp.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            self.qp.restore()

    def deactivate_area(self):
        self.activated_area_name = None



    def get_hex(self, position):
        """Return coordinates of a Hex from the given pixel position
        """
        size = self.size()
        x = size.width()//2
        y = size.height()//2
        hexgrid = hexutil.HexGrid(10)
        return hexgrid.hex_at_coordinate(position.x() - x, position.y() - y)

class ClientUI(QWidget):
    """Dice Wars' graphical user interface
    """
    def __init__(self, game):
        """
        Parameters
        ----------
        game : Game
        """
        super(ClientUI, self).__init__()
        self.game = game
        self.window_name = 'Dice Wars - Player ' + str(self.game.player_name)
        self.init_ui()


    def init_ui(self):
        self.resize(1024, 576)
        self.setMinimumSize(1024, 576)
        self.setWindowTitle(self.window_name)

        self.init_layout()
        self.show()

    def init_layout(self):
        grid = QGridLayout()

        self.main_area = MainWindow(self.game)
        self.end_turn = QPushButton('End turn')


        grid.addWidget(self.main_area, 0, 0, 10, 8)

        self.setLayout(grid)

    def handle_end_turn_button(self):
        global nb_transfers_this_turn  # dirty hack, see the top section of the module
        nb_transfers_this_turn = 0
        self.game.send_message('end_turn')



