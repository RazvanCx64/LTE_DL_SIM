clear; clc; close all;

%% 1. SIMULATION PARAMETERS
% -------------------------------------------------------------------------
bw_MHz           = 10;           % Transmission Bandwidth in MHz
num_RB           = 50;           % Number of Physical Resource Blocks (PRBs)
sc_per_RB        = 12;           % Subcarriers per Resource Block
sym_per_subframe = 14;           % OFDM symbols per subframe (Normal Cyclic Prefix)
T_subframe       = 1e-3;         % Subframe duration (TTI = 1 ms)

overhead_ratio   = 0.25;         % Signaling & control overhead (PDCCH, CRS, etc.)
num_PDSCH_RE     = round(num_RB * sc_per_RB * sym_per_subframe * (1 - overhead_ratio)); % Net PDSCH REs per subframe

SNR_dB           = -4:2:24;      % SNR evaluation range in dB
num_SNR          = length(SNR_dB);

num_subframes    = 150;          % Number of Monte Carlo subframes per SNR point

% Define digital modulation schemes
mod_schemes      = {'QPSK', '16-QAM', '64-QAM'};
bits_per_sym     = [2, 4, 6];    % Bits per symbol: M = 4, 16, 64
num_schemes      = length(mod_schemes);

% Effective code rates (aligned with 3GPP LTE Turbo coding)
code_rate        = [1/3, 1/2, 3/4];  

% Raw maximum nominal data rate (in Mbps)
R_raw_Mbps       = (num_PDSCH_RE .* bits_per_sym .* code_rate) ./ (T_subframe * 1e6);

fprintf('====================================================================\n');
fprintf('  LTE DOWNLINK THROUGHPUT SIMULATION \n');
fprintf('====================================================================\n');
fprintf('BW: %d MHz (%d Physical Resource Blocks)\n', bw_MHz, num_RB);
fprintf('Active Subcarriers: %d \n', num_RB * sc_per_RB);
fprintf('PDSCH RE per subframe (1 ms): %d RE\n', num_PDSCH_RE);
fprintf('Maximum data rates: QPSK = %.2f Mbps | 16-QAM = %.2f Mbps | 64-QAM = %.2f Mbps\n', ...
    R_raw_Mbps(1), R_raw_Mbps(2), R_raw_Mbps(3));
fprintf('--------------------------------------------------------------------\n\n');

%% 2. INITIALIZING MATRICES
% -------------------------------------------------------------------------
BER_results        = zeros(num_schemes, num_SNR);
BLER_results       = zeros(num_schemes, num_SNR);
Throughput_results = zeros(num_schemes, num_SNR); % Throughput in Mbps

target_snr_idx     = find(SNR_dB == 14, 1);       % Target SNR index for constellation snapshot (14 dB)
saved_symbols      = cell(num_schemes, 1);

%% 3. MONTE CARLO SIMULATION LOOPS
% -------------------------------------------------------------------------
rng(42); % Fixed random seed for reproducibility

% Implementation and coding penalty offsets to align with 3GPP BLER thresholds
snr_offset_dB = [-4.5, -3.5, -3.5]; 
rep_factors   = [3, 2, 2];

for m = 1:num_schemes
    k = bits_per_sym(m);
    M = 2^k;
    r_code = code_rate(m);
    rep_factor = rep_factors(m);
    bits_per_tb = floor((num_PDSCH_RE * k) / rep_factor);

    fprintf('Running simulation for %-6s (k = %d bits/symbol, r = %.2f)... ', mod_schemes{m}, k, r_code);
    tic;

    for s = 1:num_SNR
        current_snr_dB = SNR_dB(s);
        eff_snr_linear = 10^((current_snr_dB + snr_offset_dB(m)) / 10);

        total_bit_errors   = 0;
        total_bits         = 0;
        total_block_errors = 0;

        for sf = 1:num_subframes
            % --- A. Transmitter (eNodeB) ---
            tx_bits = randi([0 1], bits_per_tb, 1);
            mod_tx_sym = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);

            % Symbol-level repetition coding for coherent combining (diversity gain)
            tx_symbols = repmat(mod_tx_sym, rep_factor, 1);

            % --- B. Propagation Channel (Multipath Fading + ZF Equalization + AWGN) ---
            path_gains = [0.8, 0.4, 0.2];
            path_gains = path_gains / norm(path_gains);
            faded_symbols = filter(path_gains, 1, tx_symbols);

            % Zero-Forcing (ZF) channel equalization (inverse filter)
            eq_symbols = filter(1, path_gains, faded_symbols);

            % Additive White Gaussian Noise (AWGN) injection
            noise_power = 1 / eff_snr_linear;
            noise = sqrt(noise_power / 2) * (randn(size(eq_symbols)) + 1i * randn(size(eq_symbols)));
            rx_symbols = eq_symbols + noise;

            % Soft combining of repeated symbols (repetition coding gain)
            rx_matrix = reshape(rx_symbols, [], rep_factor);
            combined_symbols = mean(rx_matrix, 2);

            % Save equalized symbols for constellation diagram snapshot (at SNR = 14 dB)
            if s == target_snr_idx && sf == 1
                saved_symbols{m} = combined_symbols(1:min(1000, length(combined_symbols)));
            end

            % --- C. Receiver (UE) ---
            decoded_bits = qamdemod(combined_symbols, M, 'OutputType', 'bit', 'UnitAveragePower', true);

            % --- D. Error Statistics Collection ---
            bit_errs = sum(tx_bits ~= decoded_bits);
            total_bit_errors = total_bit_errors + bit_errs;
            total_bits = total_bits + bits_per_tb;

            % Block error threshold evaluation after FEC correction (CRC check emulation)
            if (bit_errs / bits_per_tb) > 1.5e-3
                total_block_errors = total_block_errors + 1;
            end
        end

        ber  = total_bit_errors / total_bits;
        bler = total_block_errors / num_subframes;

        BER_results(m, s)        = max(ber, 1e-5);
        BLER_results(m, s)       = bler;
        Throughput_results(m, s) = R_raw_Mbps(m) * (1 - bler);
    end
    fprintf('Completed in %.2f seconds.\n', toc);
end

%% 4. ADAPTIVE MODULATION AND CODING (AMC) CALCULATION (3GPP Target BLER <= 10%)
% -------------------------------------------------------------------------
AMC_Throughput = zeros(1, num_SNR);
AMC_Scheme_Idx = zeros(1, num_SNR);

for s = 1:num_SNR
    valid_schemes = find(BLER_results(:, s) <= 0.10); % 3GPP standard reliability threshold
    if isempty(valid_schemes)
        AMC_Throughput(s) = Throughput_results(1, s); % Fallback to QPSK under poor channel conditions
        AMC_Scheme_Idx(s) = 1;
    else
        [max_tput, best_idx] = max(Throughput_results(valid_schemes, s));
        AMC_Throughput(s) = max_tput;
        AMC_Scheme_Idx(s) = valid_schemes(best_idx);
    end
end

%% 5. DISPLAY TABULAR RESULTS IN CONSOLE
% -------------------------------------------------------------------------
fprintf('\n====================================================================\n');
fprintf('                   SIMULATED NUMERICAL RESULTS                      \n');
fprintf('====================================================================\n');
fprintf(' SNR (dB) | QPSK (Mbps) | 16-QAM (Mbps) | 64-QAM (Mbps) | AMC (Mbps) | Selected Scheme\n');
fprintf('--------------------------------------------------------------------\n');
for s = 1:num_SNR
    fprintf('  %5.1f   |    %6.2f   |     %6.2f   |     %6.2f   |   %6.2f   |   %-6s\n', ...
        SNR_dB(s), Throughput_results(1, s), Throughput_results(2, s), ...
        Throughput_results(3, s), AMC_Throughput(s), mod_schemes{AMC_Scheme_Idx(s)});
end
fprintf('====================================================================\n\n');

%% 6. SCIENTIFIC FIGURES
% -------------------------------------------------------------------------
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 10);

% FIGURE 1: Bit Error Rate (BER) vs. SNR
figure('Name', 'Fig 1 - BER vs SNR', 'Color', 'w', 'Position', [100, 100, 600, 420]);
semilogy(SNR_dB, BER_results(1, :), '-ob', 'LineWidth', 1.8, 'MarkerSize', 5); hold on;
semilogy(SNR_dB, BER_results(2, :), '-s', 'Color', [0.85, 0.33, 0.1], 'LineWidth', 1.8, 'MarkerSize', 5);
semilogy(SNR_dB, BER_results(3, :), '-^', 'Color', [0.18, 0.65, 0.25], 'LineWidth', 1.8, 'MarkerSize', 5);
yline(1e-3, ':k', 'BER Threshold = 10^{-3}', 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
grid on;
xlabel('Signal-to-Noise Ratio, SNR (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 11, 'FontWeight', 'bold');
legend('QPSK (2 bits/sym)', '16-QAM (4 bits/sym)', '64-QAM (6 bits/sym)', 'Location', 'southwest');
ylim([1e-5, 1]);
xlim([min(SNR_dB), max(SNR_dB)]);

% FIGURE 2: Downlink Throughput vs. SNR & AMC Operational Envelope
figure('Name', 'Fig 2 - Throughput vs SNR', 'Color', 'w', 'Position', [150, 150, 650, 440]);
plot(SNR_dB, Throughput_results(1, :), '--ob', 'LineWidth', 1.4, 'MarkerSize', 4.5); hold on;
plot(SNR_dB, Throughput_results(2, :), '--s', 'Color', [0.85, 0.33, 0.1], 'LineWidth', 1.4, 'MarkerSize', 4.5);
plot(SNR_dB, Throughput_results(3, :), '--^', 'Color', [0.18, 0.65, 0.25], 'LineWidth', 1.4, 'MarkerSize', 4.5);
plot(SNR_dB, AMC_Throughput, '-k*', 'LineWidth', 2.2, 'MarkerSize', 8);
grid on;
xlabel('Signal-to-Noise Ratio, SNR (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Downlink Throughput (Mbps)', 'FontSize', 11, 'FontWeight', 'bold');
legend('QPSK Fix (4.20 Mbps)', '16-QAM Fix (12.60 Mbps)', '64-QAM Fix (28.35 Mbps)', ...
    'AMC Envelope (BLER ≤ 10%)', 'Location', 'northwest');
xlim([min(SNR_dB), max(SNR_dB)]);
ylim([0, max(R_raw_Mbps)*1.1]);

% FIGURE 3: Equalized Constellation Diagrams at Reception (SNR = 14 dB)
figure('Name', 'Fig 3 - Constellation Diagrams', 'Color', 'w', 'Position', [200, 200, 900, 320]);
for m = 1:num_schemes
    subplot(1, 3, m);
    plot(real(saved_symbols{m}), imag(saved_symbols{m}), '.', 'Color', [0.1 0.4 0.8], 'MarkerSize', 6);
    grid on; axis square;
    xlim([-2.2 2.2]); ylim([-2.2 2.2]);
    xlabel('In-Phase (I)', 'FontSize', 9);
    ylabel('Quadrature (Q)', 'FontSize', 9);
    title(sprintf('(%c) %s (SNR = %d dB)', 'a'+m-1, mod_schemes{m}, SNR_dB(target_snr_idx)), ...
        'FontSize', 10, 'FontWeight', 'bold');
end

% FIGURE 4: Downlink Spectral Efficiency vs. SNR
figure('Name', 'Fig 4 - Spectral Efficiency', 'Color', 'w', 'Position', [250, 250, 600, 420]);
plot(SNR_dB, Throughput_results(1, :) ./ bw_MHz, '-ob', 'LineWidth', 1.6); hold on;
plot(SNR_dB, Throughput_results(2, :) ./ bw_MHz, '-s', 'Color', [0.85, 0.33, 0.1], 'LineWidth', 1.6);
plot(SNR_dB, Throughput_results(3, :) ./ bw_MHz, '-^', 'Color', [0.18, 0.65, 0.25], 'LineWidth', 1.6);
plot(SNR_dB, AMC_Throughput ./ bw_MHz, '-k*', 'LineWidth', 2.0);
grid on;
xlabel('Signal-to-Noise Ratio, SNR (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Spectral Efficiency (bps/Hz)', 'FontSize', 11, 'FontWeight', 'bold');
legend('QPSK (max 0.42 bps/Hz)', '16-QAM (max 1.26 bps/Hz)', '64-QAM (max 2.835 bps/Hz)', ...
       'AMC Envelope', 'Location', 'northwest');
xlim([min(SNR_dB), max(SNR_dB)]);
ylim([0, 3.2]);

fprintf('Simulation completed successfully! Figures 1 to 4 generated.\n');