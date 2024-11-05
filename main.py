import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

def generate_ac_signal(t, vrms=220, freq=60):
    """Gera um sinal AC senoidal"""
    vpeak = vrms * np.sqrt(2)
    return vpeak * np.sin(2 * np.pi * freq * t)

def quantize_signal(signal, n_bits, v_ref=5):
    """Quantiza o sinal para n_bits"""
    levels = 2**n_bits
    step = (2 * v_ref) / levels
    scaled_signal = (signal + v_ref) / (2 * v_ref) * levels
    quantized = np.round(scaled_signal) * step - v_ref
    return quantized

def add_conversion_delay(signal, delay_samples):
    """Adiciona atraso de conversão ao sinal"""
    return np.roll(signal, delay_samples)

def calculate_error_metrics(original, processed):
    """Calcula métricas de erro entre sinais"""
    mse = np.mean((original - processed) ** 2)
    max_error = np.max(np.abs(original - processed))
    return mse, max_error

def main():
    st.title("Análise de Conversores A/D para Medição de Tensão AC")
    
    # Parâmetros da simulação
    st.sidebar.header("Parâmetros da Simulação")
    
    # Tempo de simulação
    t = np.linspace(0, 0.1, 1000)  # 100ms com 1000 pontos
    
    # Parâmetros do sinal
    vrms = st.sidebar.number_input("Tensão RMS (V)", value=220)
    freq = st.sidebar.number_input("Frequência (Hz)", value=60)
    
    # Gerar sinal original
    signal_original = generate_ac_signal(t, vrms, freq)
    
    # Fator de escala para adequar ao range do ADC
    scale_factor = 5/311  # Escala 220Vrms para ±5V
    signal_scaled = signal_original * scale_factor
    
    # Configurações dos ADCs
    bits_options = [8, 12, 16]
    selected_bits = st.sidebar.multiselect(
        "Selecione as resoluções para análise (bits)",
        bits_options,
        default=bits_options
    )
    
    delay_samples = st.sidebar.slider("Atraso de conversão (amostras)", 0, 20, 5)
    
    # Plotar resultados
    st.header("1. Análise do Sinal no Tempo")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t*1000, signal_scaled, label='Sinal Original (Escalado)', alpha=0.8)
    
    results_df = pd.DataFrame()
    
    for bits in selected_bits:
        # Quantização
        signal_quantized = quantize_signal(signal_scaled, bits)
        
        # Adicionar atraso
        signal_delayed = add_conversion_delay(signal_quantized, delay_samples)
        
        # Plotar
        ax.plot(t*1000, signal_delayed, 
                label=f'ADC {bits} bits', 
                linestyle='--', 
                alpha=0.6)
        
        # Calcular métricas
        mse, max_error = calculate_error_metrics(signal_scaled, signal_delayed)
        
        # Adicionar resultados ao DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'Resolução (bits)': [bits],
            'MSE': [mse],
            'Erro Máximo': [max_error],
            'Atraso (amostras)': [delay_samples]
        })])
    
    ax.set_xlabel('Tempo (ms)')
    ax.set_ylabel('Tensão (V)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    # Análise de erro
    st.header("2. Análise de Erro de Quantização")
    st.dataframe(results_df)
    
    # Análise espectral
    st.header("3. Análise Espectral")
    
    fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
    
    for bits in selected_bits:
        signal_quantized = quantize_signal(signal_scaled, bits)
        signal_delayed = add_conversion_delay(signal_quantized, delay_samples)
        
        # Calcular FFT
        fft = np.fft.fft(signal_delayed)
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        
        # Plotar apenas frequências positivas até 500Hz
        mask = (freqs >= 0) & (freqs <= 500)
        ax_fft.plot(freqs[mask], 
                   2.0/len(t) * np.abs(fft[mask]), 
                   label=f'ADC {bits} bits')
    
    ax_fft.set_xlabel('Frequência (Hz)')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.grid(True)
    ax_fft.legend()
    st.pyplot(fig_fft)
    
    # Conclusões
    st.header("4. Conclusões")
    st.write("""
    Com base nas simulações realizadas, podemos observar:
    
    1. **Efeito da Resolução:**
       - Quanto maior o número de bits, menor o erro de quantização
       - O ADC de 8 bits apresenta degraus mais visíveis na forma de onda
    
    2. **Efeito do Atraso:**
       - O atraso de conversão causa uma defasagem no sinal
       - Este atraso pode afetar significativamente o desempenho de um PLL
    
    3. **Análise Espectral:**
       - Conversores com menor resolução introduzem mais ruído de quantização
       - O espectro mostra componentes de frequência adicionais devido à quantização
    """)

if __name__ == "__main__":
    main()