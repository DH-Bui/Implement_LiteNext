from tabulate import tabulate

def ExportFinalReport(history):
    history_np = np.array(history)
    best_idx = np.argmax(history_np[:, 2])
    best_stats = history[best_idx]
    result_table = [
        ["Chỉ số (Metric)", "Giá trị tốt nhất (Best)"],
        ["Dice Score", f"{best_stats[2]:.4f}"],
        ["IoU Score", f"{best_stats[3]:.4f}"],
        ["Precision", f"{best_stats[4]:.4f}"],
        ["Recall", f"{best_stats[5]:.4f}"],
        ["Training Loss", f"{best_stats[1]:.4f}"],
        ["Tại Epoch", f"{int(best_stats[0])}"]
    ]
    config_table = [
        ["Thông số", "Cấu hình"],
        ["Mô hình", "LiteNeXt (LGMNet)"],
        ["Phương pháp", "Self-Supervised (BYOL)"],
        ["Dữ liệu", "REFUGE (20% nhãn)"],
        ["Optimizer", "NAdam"],
        ["LR ban đầu", f"{args.lr}"]
    ]

    print("\n" + "═"*50)
    print("KẾT QUẢ THỰC NGHIỆM TỐI ƯU NHẤT")
    print(tabulate(result_table, headers="firstrow", tablefmt="fancy_grid"))
    print("\n" + "═"*50)
    print("\n" + "CẤU HÌNH HUẤN LUYỆN")
    print(tabulate(config_table, headers="firstrow", tablefmt="fancy_grid"))

ExportFinalReport(final_history)
