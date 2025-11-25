import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.de import DE

class RastriginProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=1,
            n_ieq_constr=0,
            xl=np.array([-5.12, -5.12]),
            xu=np.array([5.12, 5.12])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        A = 10
        f = A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        out["F"] = f

algorithm = DE(
    pop_size=50,
    variant="DE/rand/1/bin",
    CR=0.9,
    F=0.8,
    dither="vector",
    jitter=False
)

termination = get_termination("n_gen", 100)

print("Đang chạy tối ưu hóa...")
res = minimize(
    RastriginProblem(),
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

print("\n--- KẾT QUẢ ---")
print(f"Vị trí tìm được (X): {res.X}")
print(f"Giá trị tối ưu (F): {res.F} (Càng gần 0 càng tốt)")

n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])

plt.figure(figsize=(10, 6))
plt.title("Hiệu quả hội tụ của DE trên hàm Rastrigin")
plt.plot(n_evals, opt, "-o", color="blue", label="DE Best Fitness")
plt.xlabel("Số lần tính toán (Evaluations)")
plt.ylabel("Giá trị hàm mục tiêu (Log Scale)")
plt.yscale("log")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()