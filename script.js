// Complex number operations
class Complex {
    constructor(real, imag) {
        this.real = real;
        this.imag = imag;
    }
    
    add(other) {
        return new Complex(this.real + other.real, this.imag + other.imag);
    }
    
    subtract(other) {
        return new Complex(this.real - other.real, this.imag - other.imag);
    }
    
    multiply(other) {
        return new Complex(
            this.real * other.real - this.imag * other.imag,
            this.real * other.imag + this.imag * other.real
        );
    }
    
    divide(other) {
        const denom = other.real * other.real + other.imag * other.imag;
        return new Complex(
            (this.real * other.real + this.imag * other.imag) / denom,
            (this.imag * other.real - this.real * other.imag) / denom
        );
    }
    
    scale(scalar) {
        return new Complex(this.real * scalar, this.imag * scalar);
    }
    
    sqrt() {
        // Complex square root: sqrt(z) = sqrt(|z|) * exp(i*arg(z)/2)
        const mag = Math.sqrt(this.real * this.real + this.imag * this.imag);
        const arg = Math.atan2(this.imag, this.real);
        const sqrtMag = Math.sqrt(mag);
        return new Complex(
            sqrtMag * Math.cos(arg / 2),
            sqrtMag * Math.sin(arg / 2)
        );
    }
    
    exp() {
        // exp(z) = exp(real) * (cos(imag) + i*sin(imag))
        const expReal = Math.exp(this.real);
        return new Complex(
            expReal * Math.cos(this.imag),
            expReal * Math.sin(this.imag)
        );
    }
}

// Matrix exponentiation using eigenvalue decomposition (stable for 2x2 matrices)
function matrixExp(A, t) {
    // A is a 2x2 complex matrix represented as array of Complex arrays
    // For 2x2 matrix, use eigenvalue decomposition: exp(At) = V * diag(exp(λ₁t), exp(λ₂t)) * V⁻¹
    
    // Extract matrix elements
    const a = A[0][0];
    const b = A[0][1];
    const c = A[1][0];
    const d = A[1][1];
    
    // Compute trace and determinant
    const trace = a.add(d);
    const det = a.multiply(d).subtract(b.multiply(c));
    
    // Compute eigenvalues: λ = (trace ± sqrt(trace² - 4*det)) / 2
    const discriminant = trace.multiply(trace).subtract(det.scale(4));
    const sqrtDisc = discriminant.sqrt();
    
    const lambda1 = trace.add(sqrtDisc).scale(0.5);
    const lambda2 = trace.subtract(sqrtDisc).scale(0.5);
    
    // Compute exp(λ₁t) and exp(λ₂t)
    const expLambda1 = lambda1.scale(t).exp();
    const expLambda2 = lambda2.scale(t).exp();
    
    // Handle degenerate case (lambda1 ≈ lambda2)
    const diff = lambda1.subtract(lambda2);
    const diffMag = Math.sqrt(diff.real * diff.real + diff.imag * diff.imag);
    
    if (diffMag < 1e-10) {
        // Degenerate case: use exp(At) = exp(λt) * (I + (A - λI)t)
        const lambda = lambda1;
        const expLambda = expLambda1;
        const I = [
            [new Complex(1, 0), new Complex(0, 0)],
            [new Complex(0, 0), new Complex(1, 0)]
        ];
        const A_minus_lambdaI = [
            [a.subtract(lambda), b],
            [c, d.subtract(lambda)]
        ];
        const scaled = [
            [A_minus_lambdaI[0][0].scale(t), A_minus_lambdaI[0][1].scale(t)],
            [A_minus_lambdaI[1][0].scale(t), A_minus_lambdaI[1][1].scale(t)]
        ];
        const temp = [
            [I[0][0].add(scaled[0][0]), I[0][1].add(scaled[0][1])],
            [I[1][0].add(scaled[1][0]), I[1][1].add(scaled[1][1])]
        ];
        return [
            [temp[0][0].multiply(expLambda), temp[0][1].multiply(expLambda)],
            [temp[1][0].multiply(expLambda), temp[1][1].multiply(expLambda)]
        ];
    }
    
    // Non-degenerate case: compute eigenvectors and construct exp(At)
    // For 2x2, we can use a simpler approach: exp(At) = (exp(λ₁t) - exp(λ₂t))/(λ₁ - λ₂) * (A - λ₂I) + exp(λ₂t) * I
    // Or: exp(At) = exp(λ₁t) * (A - λ₂I)/(λ₁ - λ₂) + exp(λ₂t) * (λ₁I - A)/(λ₁ - λ₂)
    
    const lambdaDiff = lambda1.subtract(lambda2);
    const I = [
        [new Complex(1, 0), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(1, 0)]
    ];
    
    // Compute (A - λ₂I) and (λ₁I - A)
    const A_minus_lambda2I = [
        [a.subtract(lambda2), b],
        [c, d.subtract(lambda2)]
    ];
    const lambda1I_minus_A = [
        [lambda1.subtract(a), b.scale(-1)],
        [c.scale(-1), lambda1.subtract(d)]
    ];
    
    // exp(At) = exp(λ₁t) * (A - λ₂I)/(λ₁ - λ₂) + exp(λ₂t) * (λ₁I - A)/(λ₁ - λ₂)
    const term1 = [
        [A_minus_lambda2I[0][0].divide(lambdaDiff), A_minus_lambda2I[0][1].divide(lambdaDiff)],
        [A_minus_lambda2I[1][0].divide(lambdaDiff), A_minus_lambda2I[1][1].divide(lambdaDiff)]
    ];
    const term2 = [
        [lambda1I_minus_A[0][0].divide(lambdaDiff), lambda1I_minus_A[0][1].divide(lambdaDiff)],
        [lambda1I_minus_A[1][0].divide(lambdaDiff), lambda1I_minus_A[1][1].divide(lambdaDiff)]
    ];
    
    const result = [
        [
            term1[0][0].multiply(expLambda1).add(term2[0][0].multiply(expLambda2)),
            term1[0][1].multiply(expLambda1).add(term2[0][1].multiply(expLambda2))
        ],
        [
            term1[1][0].multiply(expLambda1).add(term2[1][0].multiply(expLambda2)),
            term1[1][1].multiply(expLambda1).add(term2[1][1].multiply(expLambda2))
        ]
    ];
    
    return result;
}

// Bloch-McConnell matrix
function blochMcConnellMatrix(k_AB, k_BA, delta_omega_A, delta_omega_B, R2_A, R2_B) {
    const omega_A = 2 * Math.PI * delta_omega_A;
    const omega_B = 2 * Math.PI * delta_omega_B;
    
    // Real part (decay + exchange)
    const L = [
        [new Complex(-R2_A, 0), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(-R2_B, 0)]
    ];
    
    // Exchange matrix
    const E = [
        [new Complex(-k_AB, 0), new Complex(k_BA, 0)],
        [new Complex(k_AB, 0), new Complex(-k_BA, 0)]
    ];
    
    // Imaginary part (precession)
    const H = [
        [new Complex(0, omega_A), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(0, omega_B)]
    ];
    
    // Combine: L + H + E
    const M = [
        [L[0][0].add(H[0][0]).add(E[0][0]), L[0][1].add(H[0][1]).add(E[0][1])],
        [L[1][0].add(H[1][0]).add(E[1][0]), L[1][1].add(H[1][1]).add(E[1][1])]
    ];
    
    return M;
}

// Simulate FID
function simulateFID(k_AB, k_BA, deltaA, deltaB, R2_A, R2_B, t_max = 0.1, n_points = 256) {
    const timePoints = [];
    const fid = [];
    
    // Initial magnetization (matching Python: pA = (k_AB + k_BA)/k_BA, M0 = [1/pA, 1/pB])
    const k_ex = k_AB + k_BA;
    const pA = k_ex > 0 ? k_ex / k_BA : 2.0;
    const pB = k_ex > 0 ? k_ex / k_AB : 2.0;
    const M0 = [
        new Complex(1 / pA, 0),
        new Complex(1 / pB, 0)
    ];
    
    const dt = t_max / (n_points - 1);
    
    for (let i = 0; i < n_points; i++) {
        const t = i * dt;
        timePoints.push(t);
        
        const M = blochMcConnellMatrix(k_AB, k_BA, deltaA, deltaB, R2_A, R2_B);
        const expMt = matrixExp(M, t);
        
        // Multiply expMt by M0
        const M_t = [
            expMt[0][0].multiply(M0[0]).add(expMt[0][1].multiply(M0[1])),
            expMt[1][0].multiply(M0[0]).add(expMt[1][1].multiply(M0[1]))
        ];
        
        // Sum contributions from both states
        fid.push(M_t[0].add(M_t[1]));
    }
    
    return { timePoints, fid };
}

// FFT implementation (using simple Cooley-Tukey FFT)
function fft(x) {
    const N = x.length;
    if (N <= 1) return x;
    
    const even = [];
    const odd = [];
    for (let i = 0; i < N; i++) {
        if (i % 2 === 0) even.push(x[i]);
        else odd.push(x[i]);
    }
    
    const evenFFT = fft(even);
    const oddFFT = fft(odd);
    
    const result = new Array(N);
    for (let k = 0; k < N / 2; k++) {
        const t = new Complex(
            Math.cos(-2 * Math.PI * k / N),
            Math.sin(-2 * Math.PI * k / N)
        );
        const term = t.multiply(oddFFT[k]);
        result[k] = evenFFT[k].add(term);
        result[k + N / 2] = evenFFT[k].add(term.scale(-1));
    }
    
    return result;
}

function fftshift(arr) {
    const n = arr.length;
    const mid = Math.floor(n / 2);
    return [...arr.slice(mid), ...arr.slice(0, mid)];
}

// Compute spectrum from FID
function computeSpectrum(fid, timePoints) {
    // Apply cosine window
    const t_max = timePoints[timePoints.length - 1];
    const w = timePoints.map(t => Math.cos((t / t_max) * (Math.PI / 2)));
    const fid_w = fid.map((val, i) => 
        new Complex(val.real * w[i], val.imag * w[i])
    );
    fid_w[0] = new Complex(fid[0].real * 0.5, fid[0].imag * 0.5);
    
    // Zero fill
    const zeros = Array.from({ length: fid_w.length }, () => new Complex(0, 0));
    const fid_w_zf = [...fid_w, ...zeros];
    
    // FFT
    const spectrum = fft(fid_w_zf);
    const spectrumShifted = fftshift(spectrum);
    
    // Frequency axis - must match NumPy's fftfreq order: [0, 1, ..., N/2-1, -N/2, -N/2+1, ..., -1] * fs/N
    // Then fftshift reorders to: [-N/2, ..., -1, 0, 1, ..., N/2-1] * fs/N
    const dt = timePoints[1] - timePoints[0];
    const N = 2 * timePoints.length;
    const fs = 1.0 / dt;  // Sampling frequency
    
    // Create frequency axis in FFT output order (before fftshift)
    const freqAxis = [];
    for (let i = 0; i < N; i++) {
        if (i < N / 2) {
            // Positive frequencies: 0, 1, 2, ..., N/2-1
            freqAxis.push(i * fs / N);
        } else {
            // Negative frequencies: -N/2, -N/2+1, ..., -1
            freqAxis.push((i - N) * fs / N);
        }
    }
    
    // Apply fftshift to match the shifted spectrum
    const freqAxisShifted = fftshift(freqAxis);
    
    return { freqAxis: freqAxisShifted, spectrum: spectrumShifted };
}

// Update plots
let spectrumPlot, fidPlot;

function updatePlots() {
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not available');
        return;
    }
    
    try {
        const k_AB = Math.pow(10, parseFloat(document.getElementById('kAB').value));
        const equalK = document.getElementById('equalK').checked;
        const k_BA = equalK ? k_AB : Math.pow(10, parseFloat(document.getElementById('kBA').value));
        const R2_A = Math.pow(10, parseFloat(document.getElementById('R2A').value));
        const R2_B = Math.pow(10, parseFloat(document.getElementById('R2B').value));
        const deltaA = parseFloat(document.getElementById('deltaA').value);
        const deltaB = parseFloat(document.getElementById('deltaB').value);
        const ftMin = parseFloat(document.getElementById('ftMin').value);
        const ftMax = parseFloat(document.getElementById('ftMax').value);
        
        // Update value displays
        document.getElementById('kAB-value').textContent = k_AB.toFixed(2);
        document.getElementById('kBA-value').textContent = k_BA.toFixed(2);
        document.getElementById('R2A-value').textContent = R2_A.toFixed(2);
        document.getElementById('R2B-value').textContent = R2_B.toFixed(2);
        document.getElementById('deltaA-value').textContent = deltaA.toFixed(0);
        document.getElementById('deltaB-value').textContent = deltaB.toFixed(0);
        document.getElementById('ftMin-value').textContent = ftMin.toFixed(1);
        document.getElementById('ftMax-value').textContent = ftMax.toFixed(1);
        
        // Simulate
        const { timePoints, fid } = simulateFID(k_AB, k_BA, deltaA, deltaB, R2_A, R2_B);
        const { freqAxis, spectrum } = computeSpectrum(fid, timePoints);
        
        // Calculate probabilities
        const k_ex = k_AB + k_BA;
        const p_A = k_ex > 0 ? k_BA / k_ex : 0.5;
        const p_B = k_ex > 0 ? k_AB / k_ex : 0.5;
        const deltaDiff = Math.abs(deltaA - deltaB);
        
        // Prepare data for plots
        const spectrumReal = spectrum.map(c => c.real);
        const fidReal = fid.map(c => c.real);
        const fidImag = fid.map(c => c.imag);
        
        // Update spectrum plot
        const spectrumTrace = {
            x: freqAxis,
            y: spectrumReal,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#667eea', width: 2 },
            name: 'Spectrum'
        };
        
        const ruleA = {
            x: [deltaA, deltaA],
            y: [ftMin, ftMax],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'red', dash: 'dash', width: 2 },
            name: `ΔΩ<sub>A</sub> (p<sub>A</sub>=${p_A.toFixed(2)})`,
            showlegend: true
        };
        
        const ruleB = {
            x: [deltaB, deltaB],
            y: [ftMin, ftMax],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'blue', dash: 'dash', width: 2 },
            name: `ΔΩ<sub>B</sub> (p<sub>B</sub>=${p_B.toFixed(2)})`,
            showlegend: true
        };
        
        const spectrumLayout = {
            title: `NMR Spectrum with Two-State Exchange<br>|ΔΩ<sub>A</sub> - ΔΩ<sub>B</sub>| = ${deltaDiff.toFixed(0)} Hz, k<sub>ex</sub> = ${k_ex.toFixed(0)} Hz`,
            xaxis: { title: 'Frequency (Hz)', autorange: true },
            yaxis: { title: 'Intensity', range: [ftMin, ftMax] },
            height: 400,
            margin: { l: 60, r: 20, t: 60, b: 50 },
            hovermode: 'closest',
            showlegend: true
        };
        
        Plotly.react('spectrum-plot', [spectrumTrace, ruleA, ruleB], spectrumLayout, {responsive: true, displayModeBar: true});
        
        // Update FID plot
        const fidRealTrace = {
            x: timePoints,
            y: fidReal,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#667eea', width: 2 },
            name: 'Real'
        };
        
        const fidImagTrace = {
            x: timePoints,
            y: fidImag,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#764ba2', width: 2 },
            name: 'Imaginary'
        };
        
        const fidLayout = {
            title: 'Free Induction Decay (Real & Imaginary)',
            xaxis: { title: 'Time (s)', autorange: true },
            yaxis: { title: 'FID', autorange: true },
            height: 400,
            margin: { l: 60, r: 20, t: 40, b: 50 },
            hovermode: 'closest',
            showlegend: true
        };
        
        Plotly.react('fid-plot', [fidRealTrace, fidImagTrace], fidLayout, {responsive: true, displayModeBar: true});
    } catch (error) {
        console.error('Error updating plots:', error);
    }
}

// Set up event listeners when DOM is ready and Plotly is loaded
function initialize() {
    if (typeof Plotly === 'undefined') {
        console.log('Waiting for Plotly to load...');
        setTimeout(initialize, 100);
        return;
    }
    
    // Check if required elements exist
    const spectrumPlotDiv = document.getElementById('spectrum-plot');
    const fidPlotDiv = document.getElementById('fid-plot');
    
    if (!spectrumPlotDiv || !fidPlotDiv) {
        console.error('Plot containers not found');
        return;
    }
    
    try {
        // Flag to prevent infinite loops when syncing sliders
        let syncingSliders = false;
        
        document.getElementById('kAB').addEventListener('input', function() {
            const equalK = document.getElementById('equalK').checked;
            if (equalK && !syncingSliders) {
                syncingSliders = true;
                document.getElementById('kBA').value = this.value;
                syncingSliders = false;
            }
            updatePlots();
        });
        
        document.getElementById('kBA').addEventListener('input', function() {
            const equalK = document.getElementById('equalK').checked;
            if (equalK && !syncingSliders) {
                syncingSliders = true;
                document.getElementById('kAB').value = this.value;
                syncingSliders = false;
            }
            updatePlots();
        });
        
        document.getElementById('R2A').addEventListener('input', updatePlots);
        document.getElementById('R2B').addEventListener('input', updatePlots);
        document.getElementById('deltaA').addEventListener('input', updatePlots);
        document.getElementById('deltaB').addEventListener('input', updatePlots);
        document.getElementById('ftMin').addEventListener('input', updatePlots);
        document.getElementById('ftMax').addEventListener('input', updatePlots);
        document.getElementById('equalK').addEventListener('change', function() {
            if (this.checked) {
                document.getElementById('kBA').value = document.getElementById('kAB').value;
            }
            updatePlots();
        });
        
        console.log('Initializing plots...');
        // Initial plot
        updatePlots();
        console.log('Plots initialized');
    } catch (error) {
        console.error('Error initializing:', error);
        console.error(error.stack);
    }
}

// Wait for both DOM and Plotly to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    // DOM is already ready, but wait for Plotly
    initialize();
}

