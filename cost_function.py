import numpy as np


def compute_cost(state, control, goal, prev_state=None, params=None, world=None):
    """
    =====================================================================
    FUNCTION: compute_cost
    =====================================================================
    Opis:
        Funkcja oblicza wartość kosztu (lub negatywnej nagrody)
        dla jednego kroku symulacji robota mobilnego samobalansującego.
        Koszt reprezentuje "jakość" aktualnego ruchu robota – im mniejszy
        koszt, tym lepsze zachowanie (szybsze, stabilniejsze, bardziej
        energooszczędne).

    ---------------------------------------------------------------------
    WEJŚCIA:
    ---------------------------------------------------------------------
    state : list lub numpy.ndarray o długości 5
        Aktualny stan robota w danym kroku czasowym:
            state = [x, y, theta, phi, dphi]
            gdzie:
                x, y   – pozycja robota w przestrzeni świata [m]
                theta  – kąt orientacji robota (radiany)
                phi    – kąt pochyłu (balansu) robota [radiany]
                dphi   – prędkość kątowa pochyłu [rad/s]

    control : list lub numpy.ndarray o długości 2
        Wektor sterowania robota:
            control = [accel, steering]
            gdzie:
                accel    – przyspieszenie liniowe [m/s²]
                steering – kąt skrętu kół (lub sygnał kierowania) [radiany]

    goal : list lub numpy.ndarray o długości 2
        Punkt docelowy w przestrzeni:
            goal = [x_goal, y_goal]
            Celem robota jest minimalizacja odległości pomiędzy (x, y)
            a (x_goal, y_goal).

    prev_state : list lub numpy.ndarray, opcjonalnie (domyślnie None)
        Stan robota w poprzednim kroku czasowym.
        Używany do obliczenia postępu (przebytej odległości).
        Jeśli None – przyjmuje, że to pierwszy krok i postęp = 0.

    params : dict, opcjonalnie (domyślnie None)
        Słownik zawierający wagi poszczególnych składników funkcji kosztu.
        Dostępne klucze i wartości domyślne:
            {
                "w_stab": 10,        # waga stabilności
                "w_energy": 0.5,     # waga energii
                "w_speed": 2,        # waga prędkości / progresu
                "w_goal": 5,         # waga zbliżania się do celu
                "w_collision": 50    # kara za kolizję
            }

    world : obiekt klasy World (opcjonalnie)
        Świat, w którym porusza się robot.
        Jeśli przekazany, funkcja sprawdzi kolizję robota z przeszkodami
        za pomocą metody `world.check_collision([x, y])`.

    ---------------------------------------------------------------------
    WYJŚCIE:
    ---------------------------------------------------------------------
    cost : float
        Łączna wartość kosztu w danym kroku symulacji.
        Im mniejsza wartość, tym lepsze zachowanie robota.
        Jeśli używasz tego w systemie uczącym się (np. RL), możesz
        odwrócić znak, by uzyskać nagrodę: reward = -cost.

    =====================================================================
    WZORY I SKŁADNIKI FUNKCJI KOSZTU:
    =====================================================================

        J_stab  – koszt stabilności      → φ² + 0.1·φ̇²
        J_energy – koszt energii         → accel² + 0.1·steering²
        J_speed  – koszt (ujemny) prędkości / progresu
                    → -‖Δx, Δy‖  (nagradzamy ruch naprzód)
        J_goal   – kara za odległość od celu
                    → ||[x_goal - x, y_goal - y]||
        J_collision – kara za kolizję (duża stała, jeśli nastąpiła)

        Łączny koszt:
            cost = w_stab*J_stab
                 + w_energy*J_energy
                 + w_speed*J_speed
                 + w_goal*J_goal
                 + w_collision*J_collision
    =====================================================================
    """

    # ------------------------------------------------------------------
    # 0. Bezpieczne wartości domyślne
    # ------------------------------------------------------------------
    if params is None:
        params = {}
    if prev_state is None:
        prev_state = state

    # ------------------------------------------------------------------
    # 1. Rozpakowanie stanu i sterowania
    # ------------------------------------------------------------------
    x, y, theta, phi, dphi = state           # aktualny stan
    accel, steering = control                # sterowanie

    # ------------------------------------------------------------------
    # 2. Składnik STABILNOŚCI – chcemy, żeby robot nie tracił równowagi
    # ------------------------------------------------------------------
    # φ = kąt pochyłu, dφ = prędkość kątowa pochyłu
    # Wartość φ ≈ 0 oznacza pionową pozycję robota.
    J_stab = phi**2 + 0.1 * dphi**2

    # ------------------------------------------------------------------
    # 3. Składnik ENERGETYCZNY – im większe sterowania, tym większe zużycie
    # ------------------------------------------------------------------
    # Uproszczony model: energia ∝ kwadrat przyspieszenia i skrętu
    J_energy = accel**2 + 0.1 * steering**2

    # ------------------------------------------------------------------
    # 4. Składnik PRĘDKOŚCI / PROGRESU
    # ------------------------------------------------------------------
    # Nagroda za ruch do przodu — obliczamy dystans między stanami
    dx = x - prev_state[0]
    dy = y - prev_state[1]
    progress = np.sqrt(dx**2 + dy**2)
    # Ujemny znak, bo większy progres = mniejszy koszt
    J_speed = -progress

    # ------------------------------------------------------------------
    # 5. Składnik CELU – kara za odległość od punktu docelowego
    # ------------------------------------------------------------------
    goal_dist = np.linalg.norm([goal[0] - x, goal[1] - y])
    J_goal = goal_dist

    # ------------------------------------------------------------------
    # 6. Kolizje – duża kara jeśli robot wjedzie w przeszkodę
    # ------------------------------------------------------------------
    J_collision = 0
    if world is not None and world.check_collision([x, y]):
        J_collision = 1000  # duża kara; można zmienić na parametryzowaną

    # ------------------------------------------------------------------
    # 7. Wagi składników
    # ------------------------------------------------------------------
    w_stab = params.get("w_stab", 10)
    w_energy = params.get("w_energy", 0.5)
    w_speed = params.get("w_speed", 2)
    w_goal = params.get("w_goal", 5)
    w_collision = params.get("w_collision", 50)

    # ------------------------------------------------------------------
    # 8. Łączny koszt
    # ------------------------------------------------------------------
    cost = (
        w_stab * J_stab +
        w_energy * J_energy +
        w_speed * J_speed +
        w_goal * J_goal +
        w_collision * J_collision
    )

    return cost
