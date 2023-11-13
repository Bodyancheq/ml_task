import pygame
import numpy as np
import random

from sklearn import svm


def generate_random_points(coord):
    count = random.randint(2, 5)
    X = [coord, ]
    for i in range(count):
        angle = np.pi * random.randint(0, 360) / 180
        radius = random.randint(10, 20)
        x = radius * np.cos(angle) + coord[0]
        y = radius * np.sin(angle) + coord[1]
        X.append([x, y])
    return np.array(X)


def redraw_all():
    screen.fill(color='#FFFFFF')
    for x, y in zip(X_train, y_train):
        pygame.draw.circle(screen, color=colors[(1 + int(y)) // 2],
                           center=x, radius=r)
    for x, y in zip(X_test, y_test):
        pygame.draw.circle(screen, color=colors[(1 + int(y)) // 2],
                           center=x, radius=r)
    if GAME_MODE == LINEAR and MODEL_STATE == TRAINED:
        W = clf.coef_[0]
        I = clf.intercept_
        a = -W[0] / W[1]
        b = I[0] / W[1]
        x1 = 0
        y1 = a * x1 - b
        x2 = pygame.display.get_window_size()[0]
        y2 = a * x2 - b
        print(y1, y2)
        pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 3)


if __name__ == '__main__':
    LINEAR = 1
    NON_LINEAR = 2
    MAIN_MENU = 0
    UNTRAINED = 0
    TRAINED = 1
    GAME_MODE = MAIN_MENU
    MODEL_STATE = UNTRAINED
    r = 3
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill(color='#FFFFFF')
    pygame.display.update()
    is_pressed_l = False
    is_pressed_m = False
    is_pressed_r = False
    X_train = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    X_test = np.array([])
    colors = ["#FF0000", "#0000FF", "#00FF00"]

    clf = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.VIDEORESIZE:
                redraw_all()

            if GAME_MODE == MAIN_MENU:
                img = pygame.font.SysFont('arialblack', 40).render("Press 1 for linear kernel and 2 for non-linear.",
                                                                   True, (0, 0, 0,))
                screen.blit(img, (50, 50))
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_1:
                        GAME_MODE = LINEAR
                        screen.fill(color='#FFFFFF')
                        clf = svm.SVC(kernel='linear')
                    if event.key == pygame.K_2:
                        GAME_MODE = NON_LINEAR
                        screen.fill(color='#FFFFFF')
                        clf = svm.SVC()

                pygame.display.update()
                continue
            else:
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        GAME_MODE = MAIN_MENU
                        MODEL_STATE = UNTRAINED
                        screen.fill(color='#FFFFFF')
                        X_train = np.array([])
                        X_test = np.array([])
                        y_train = np.array([])
                        y_test = np.array([])
            print(event)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_BACKSPACE:
                    screen.fill(color='#FFFFFF')
                    X_train = np.array([])
                    X_test = np.array([])
                    y_train = np.array([])
                    y_test = np.array([])
                    MODEL_STATE = UNTRAINED
                elif event.key == 13:
                    if X_train.size:
                        clf.fit(X_train, y_train)
                        MODEL_STATE = TRAINED
                        if X_test.size:
                            y_test = clf.predict(X_test)
                        redraw_all()
                        if GAME_MODE == LINEAR:
                            W = clf.coef_[0]
                            I = clf.intercept_
                            a = -W[0] / W[1]
                            b = I[0] / W[1]
                            x1 = 0
                            y1 = a * x1 - b
                            x2 = pygame.display.get_window_size()[0]
                            y2 = a * x2 - b
                            print(y1, y2)
                            pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 3)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed_l = True
                elif event.button == 2:
                    is_pressed_m = True
                elif event.button == 3:
                    is_pressed_r = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_pressed_l = False
                if event.button == 2:
                    is_pressed_m = False
                if event.button == 3:
                    is_pressed_r = False
            if is_pressed_m:
                coord = np.array(event.pos)
                if X_test.size:
                    if np.linalg.norm(X_test[-1] - coord) > 5 * r:
                        near_points = generate_random_points(coord)
                        X_test = np.concatenate((X_test, near_points))
                        if MODEL_STATE == TRAINED:
                            new_y = clf.predict(near_points)
                        else:
                            new_y = np.array([3] * len(near_points))
                        y_test = np.concatenate((y_test, new_y))
                        for x, y in zip(near_points, new_y):
                            pygame.draw.circle(screen, color=colors[(1 + int(y)) // 2],
                                               center=x, radius=r)
                else:
                    X_test = np.array([coord, ])
                    if MODEL_STATE == TRAINED:
                        y_test = clf.predict(X_test)
                    else:
                        y_test = np.array([3])
                    pygame.draw.circle(screen, color=colors[(1 + int(y_test[0])) // 2],
                                       center=coord, radius=r)

            if is_pressed_l or is_pressed_r:
                y_true = -1 if is_pressed_l else 1
                color = colors[(1 + y_true) // 2]
                coord = np.array(event.pos)
                if X_train.size:
                    if np.linalg.norm(X_train[-1] - coord) > 5 * r:
                        near_points = generate_random_points(coord)
                        X_train = np.concatenate((X_train, near_points))
                        new_y = []
                        for elem in near_points:
                            pygame.draw.circle(screen, color=color,
                                               center=elem, radius=r)
                            new_y.append(y_true)
                        y_train = np.concatenate((y_train, np.array(new_y)))
                else:
                    X_train = np.array([coord, ])
                    new_y = []
                    pygame.draw.circle(screen, color=color,
                                       center=coord, radius=r)
                    new_y.append(y_true)
                    y_train = np.concatenate((y_train, np.array(new_y)))

            pygame.display.update()
