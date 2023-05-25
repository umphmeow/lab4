
from matplotlib import pyplot as plt
import numpy as np
def calculate_expression(A, F, k):
    trans_A = np.transpose(A)
    trans_F = np.transpose(F)
    det_A = np.linalg.det(A)
    diag_F = np.trace(F)

    if det_A > diag_F:
        print('Вычисляем выражение : A-1*AT – K * F-1 ')
        print('Транспонированая матрица A:\n', trans_A)

        power_A = np.linalg.matrix_power(A, -1)
        print('Возведение матрицы A в -1 степень:\n', power_A)

        mod_A = np.dot(A, trans_A)
        print('Умножение A-1 * AT\n', mod_A)

        power_F = np.linalg.matrix_power(F, -1)
        print('Возведение матрицы F в -1 степень:\n', power_F)

        mod_power_F = np.dot(k, power_F)
        print('Умножение K *F-1\n', mod_power_F)

        result = np.subtract(mod_A, mod_power_F)
        print('Разница матриц\n', result)
    else:
        print('Вычисляем выражение: (AТ + G-FТ)*K ')
        print('Транспонированая матрица F:\n', trans_F)

        G = np.tril(A)
        print('Нижняя треугольная матрица G из матрицы A:\n', G)

        pAG = np.subtract(trans_F, G)
        print('Разница A-1 - G:\n', pAG)

        print('Транспонированая матрица A:\n', trans_A)

        pAGFT = np.subtract(pAG, trans_F)
        print('Разница At + G - FT:\n', pAGFT)

        result = np.dot(pAGFT, k)
        print('Умножение на K\n', result)

    print('Результат вычислений\n', result)
    return result

#######################################################################################################################

k = int(input("Введите число K  которое является коэффициентом при умножении: "))
n = int(input("Введите размерность матрицы A(чётное): "))
while n <= 3 or n % 2 != 0:
    n = int(input("Введите число больше 3: "))
A = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        A[i][j] = A[j][i] = np.random.randint(-10, 11)
print("Матрица A:\n", A)

half_n = n // 2
maxfix_n = half_n
minfix_n = half_n
if n % 2 != 0:
    maxfix_n += 1
    minfix_n = maxfix_n - 1

D = A[:minfix_n, :minfix_n]
print("Подматрица D:\n", D)

E = A[:minfix_n, maxfix_n:]
print("Подматрица E:\n", E)

B = A[maxfix_n:, maxfix_n:]
print("Подматрица B:\n", B)

C = A[maxfix_n:, :minfix_n]
print("Подматрица C:\n", C)

#######################################################################################################################

if np.array_equal(A, A.T):
    print("Матрица A симметрична относительно главной диагонали")
    temp = np.fliplr(C)
    C = np.fliplr(B)
    B = np.copy(temp)
    print("Подматрица B после замены:\n", B)
    print("Подматрица C после замены:\n", C)
else:
    print("Матрица A не симметрична относительно главной диагонали")
    temp = np.copy(C)
    C = np.copy(E)
    E = np.copy(temp)
    print("Подматрица C после замены:\n", C)
    print("Подматрица E после замены:\n", E)

F = np.vstack((np.hstack((D, E)), np.hstack((C, B))))
print("Матрица F:\n", F)

print(calculate_expression(A, F, k))

#######################################################################################################################

explode = [0] * (n - 1)
explode.append(0.1)
plt.title("Круговая диаграмма")
try:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
except IndexError:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
plt.pie(sizes, labels=list(range(1, n + 1)), explode=explode, autopct='%1.1f%%', shadow=True)
plt.show()

plt.plot(A)
plt.title("График")
plt.ylabel("y axis")
plt.xlabel("x axis")
plt.show()

# Построение тепловой карты
fig, ax = plt.subplots()
im = ax.imshow(A, cmap='coolwarm')

# Добавление подписей для осей
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(range(1, n+1))
ax.set_yticklabels(range(1, n+1))
plt.xlabel("Номер столбца")
plt.ylabel("Номер строки")

# Добавление аннотаций в ячейки
for i in range(n):
    for j in range(n):
        text = ax.text(j, i, A[i, j], ha="center", va="center", color="black")

# Добавление цветовой шкалы
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Значение", rotation=-90, va="bottom")

# Добавление заголовка
plt.title("Тепловая карта матрицы A")

plt.show()
