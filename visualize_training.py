import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(csv_path='./checkpoints/training_history.csv'):
    """Построение графиков обучения"""
    
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден!")
        print("Сначала обучите модель.")
        return
    
    # Загрузка данных
    df = pd.read_csv(csv_path)
    epochs = range(1, len(df) + 1)
    
    # Создание фигуры с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Результаты обучения DeepSpeech2', fontsize=16, fontweight='bold')
    
    # 1. Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, df['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. WER
    ax2 = axes[0, 1]
    ax2.plot(epochs, np.array(df['test_wer']) * 100, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('WER (%)', fontsize=12)
    ax2.set_title('Word Error Rate (WER)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Добавление минимального значения
    min_wer_idx = df['test_wer'].idxmin()
    min_wer = df['test_wer'].iloc[min_wer_idx]
    ax2.axhline(y=min_wer*100, color='r', linestyle='--', alpha=0.5, 
                label=f'Best: {min_wer*100:.2f}%')
    ax2.legend(fontsize=10)
    
    # 3. CER
    ax3 = axes[1, 0]
    ax3.plot(epochs, np.array(df['test_cer']) * 100, 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('CER (%)', fontsize=12)
    ax3.set_title('Character Error Rate (CER)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Добавление минимального значения
    min_cer_idx = df['test_cer'].idxmin()
    min_cer = df['test_cer'].iloc[min_cer_idx]
    ax3.axhline(y=min_cer*100, color='r', linestyle='--', alpha=0.5,
                label=f'Best: {min_cer*100:.2f}%')
    ax3.legend(fontsize=10)
    
    # 4. Сравнение WER и CER
    ax4 = axes[1, 1]
    ax4.plot(epochs, np.array(df['test_wer']) * 100, 'g-', 
             label='WER', linewidth=2)
    ax4.plot(epochs, np.array(df['test_cer']) * 100, 'm-', 
             label='CER', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Error Rate (%)', fontsize=12)
    ax4.set_title('WER vs CER', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение графика
    output_path = './checkpoints/training_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_path}")
    
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*60)
    print("СТАТИСТИКА ОБУЧЕНИЯ")
    print("="*60)
    print(f"\nВсего эпох: {len(df)}")
    print(f"\nФинальные значения:")
    print(f"  Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Test Loss:  {df['test_loss'].iloc[-1]:.4f}")
    print(f"  WER:        {df['test_wer'].iloc[-1]*100:.2f}%")
    print(f"  CER:        {df['test_cer'].iloc[-1]*100:.2f}%")
    
    print(f"\nЛучшие значения:")
    print(f"  Best WER:   {min_wer*100:.2f}% (эпоха {min_wer_idx+1})")
    print(f"  Best CER:   {min_cer*100:.2f}% (эпоха {min_cer_idx+1})")
    print(f"  Best Train Loss: {df['train_loss'].min():.4f}")
    print(f"  Best Test Loss:  {df['test_loss'].min():.4f}")
    
    print("\n" + "="*60)


def plot_single_metric(csv_path='./checkpoints/training_history.csv', 
                       metric='test_wer', 
                       title='Word Error Rate'):
    """Построение графика одной метрики"""
    
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден!")
        return
    
    df = pd.read_csv(csv_path)
    epochs = range(1, len(df) + 1)
    
    plt.figure(figsize=(10, 6))
    
    if metric in ['test_wer', 'test_cer']:
        values = np.array(df[metric]) * 100
        ylabel = 'Error Rate (%)'
    else:
        values = df[metric]
        ylabel = 'Loss'
    
    plt.plot(epochs, values, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Минимальное значение
    min_idx = np.argmin(values)
    min_val = values[min_idx]
    plt.axhline(y=min_val, color='r', linestyle='--', alpha=0.5,
                label=f'Best: {min_val:.2f}')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_path = f'./checkpoints/{metric}_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_path}")
    
    plt.show()


def compare_models(csv_paths, labels):
    """Сравнение нескольких моделей"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Сравнение моделей', fontsize=16, fontweight='bold')
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        if not os.path.exists(csv_path):
            print(f"Файл {csv_path} не найден, пропускаю...")
            continue
        
        df = pd.read_csv(csv_path)
        epochs = range(1, len(df) + 1)
        color = colors[i % len(colors)]
        
        # WER
        axes[0].plot(epochs, np.array(df['test_wer']) * 100, 
                     color=color, label=label, linewidth=2)
        
        # Loss
        axes[1].plot(epochs, df['test_loss'], 
                     color=color, label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('WER (%)', fontsize=12)
    axes[0].set_title('Word Error Rate', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = './checkpoints/models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сравнения сохранен: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Визуализация результатов обучения')
    parser.add_argument('--csv', type=str, 
                        default='./checkpoints/training_history.csv',
                        help='Путь к CSV файлу с историей')
    parser.add_argument('--metric', type=str, 
                        default=None,
                        help='Конкретная метрика для отображения (test_wer, test_cer, etc.)')
    parser.add_argument('--compare', nargs='+', 
                        default=None,
                        help='Пути к нескольким CSV для сравнения')
    parser.add_argument('--labels', nargs='+',
                        default=None,
                        help='Метки для сравниваемых моделей')
    
    args = parser.parse_args()
    
    if args.compare and args.labels:
        # Сравнение моделей
        compare_models(args.compare, args.labels)
    elif args.metric:
        # Отображение одной метрики
        plot_single_metric(args.csv, args.metric)
    else:
        # Отображение всех метрик
        plot_training_history(args.csv)
