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
            line: { color: '#7B2D3B', width: 2 },
            name: 'Spectrum'
        };
        
        const ruleA = {
            x: [deltaA, deltaA],
            y: [ftMin, ftMax],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#A3445A', dash: 'dash', width: 2 },
            name: `ΔΩ<sub>A</sub> (p<sub>A</sub>=${p_A.toFixed(2)})`,
            showlegend: true
        };
        
        const ruleB = {
            x: [deltaB, deltaB],
            y: [ftMin, ftMax],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#D4764E', dash: 'dash', width: 2 },
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
            line: { color: '#7B2D3B', width: 2 },
            name: 'Real'
        };
        
        const fidImagTrace = {
            x: timePoints,
            y: fidImag,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#E0829B', width: 2 },
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

// Set up event listeners for BM tab (lazy-initialised on first switch)
function initSimulation() {
    if (typeof Plotly === 'undefined') {
        setTimeout(initSimulation, 100);
        return;
    }
    
    // Check if required elements exist
    const spectrumPlotDiv = document.getElementById('spectrum-plot');
    const fidPlotDiv = document.getElementById('fid-plot');
    
    if (!spectrumPlotDiv || !fidPlotDiv) {
        console.error('Plot containers not found');
        return;
    }
    
    simulationInitialized = true;

    try {
        // Flag to prevent infinite loops when syncing sliders
        let syncingSliders = false;
        // When "lock ratio" is checked, log10(k_BA/k_AB) = kBA_slider - kAB_slider
        let lockedLogRatio = 0;

        document.getElementById('kAB').addEventListener('input', function() {
            const equalK = document.getElementById('equalK').checked;
            const lockRatio = document.getElementById('lockRatio').checked;
            if (equalK && !syncingSliders) {
                syncingSliders = true;
                document.getElementById('kBA').value = this.value;
                syncingSliders = false;
            } else if (lockRatio && !syncingSliders) {
                syncingSliders = true;
                const v = Math.max(-1, Math.min(6, parseFloat(this.value) + lockedLogRatio));
                document.getElementById('kBA').value = v;
                syncingSliders = false;
            }
            updatePlots();
        });

        document.getElementById('kBA').addEventListener('input', function() {
            const equalK = document.getElementById('equalK').checked;
            const lockRatio = document.getElementById('lockRatio').checked;
            if (equalK && !syncingSliders) {
                syncingSliders = true;
                document.getElementById('kAB').value = this.value;
                syncingSliders = false;
            } else if (lockRatio && !syncingSliders) {
                syncingSliders = true;
                const v = Math.max(-1, Math.min(6, parseFloat(this.value) - lockedLogRatio));
                document.getElementById('kAB').value = v;
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
        document.getElementById('lockRatio').addEventListener('change', function() {
            if (this.checked) {
                const kAB = document.getElementById('kAB');
                const kBA = document.getElementById('kBA');
                lockedLogRatio = parseFloat(kBA.value) - parseFloat(kAB.value);
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

// ============================================================
// Bloch Equations tab (introductory – no exchange)
// ============================================================
let blochInitialized = false;
let blochVectorInitialized = false;
let blochVectorAnimationId = null;
let blochVectorTime = 0;
let blochVectorLastTs = null;
const blochVectorTraceIndex = 3;
const blochVectorPathIndex = 4;
let blochVectorPathX = [];
let blochVectorPathY = [];
let blochVectorPathZ = [];
const blochVectorMaxStep = 0.002;
let blochPrePulseMz = 1;
let blochMz0 = 0;
let blochMxyScale = 1;

function simulateBlochFID(delta, R2, flipAngleDeg, t_max = 1.0, n_points = 256) {
    const alpha = flipAngleDeg * Math.PI / 180;
    const sinAlpha = Math.sin(alpha);
    const omega = 2 * Math.PI * delta;
    const timePoints = [];
    const fid = [];
    const dt = t_max / (n_points - 1);

    for (let i = 0; i < n_points; i++) {
        const t = i * dt;
        timePoints.push(t);
        const decay = sinAlpha * Math.exp(-R2 * t);
        const re = Math.cos(omega * t) * decay;
        const im = Math.sin(omega * t) * decay;
        fid.push(new Complex(re, im));
    }
    return { timePoints, fid };
}

function getBlochVectorParams() {
    const R2 = Math.pow(10, parseFloat(document.getElementById('bloch-R2').value));
    const R1 = Math.pow(10, parseFloat(document.getElementById('bloch-R1').value));
    const delta = parseFloat(document.getElementById('bloch-delta').value);
    const flipAngle = parseFloat(document.getElementById('bloch-flip').value);
    const speed = Math.pow(10, parseFloat(document.getElementById('bloch-speed').value));
    const d1Slider = document.getElementById('bloch-d1');
    const d1EnabledEl = document.getElementById('bloch-d1-enabled');
    const d1 = d1Slider ? parseFloat(d1Slider.value) : 1.0;
    const d1Enabled = d1EnabledEl ? d1EnabledEl.checked : false;
    return { R2, R1, delta, flipAngle, speed, d1, d1Enabled };
}

function computeBlochVector(t, R2, R1, delta) {
    const omega = 2 * Math.PI * delta;
    const decay = Math.exp(-R2 * t);
    const Mx = blochMxyScale * decay * Math.cos(omega * t);
    const My = blochMxyScale * decay * Math.sin(omega * t);
    const M0 = 1;
    const Mz = M0 - (M0 - blochMz0) * Math.exp(-R1 * t);
    return { Mx, My, Mz };
}

function computeMzAtTime(t, R1) {
    const M0 = 1;
    return M0 - (M0 - blochMz0) * Math.exp(-R1 * t);
}

function resetBlochPulseState() {
    const flipAngle = parseFloat(document.getElementById('bloch-flip').value);
    const alpha = flipAngle * Math.PI / 180;
    blochMz0 = blochPrePulseMz * Math.cos(alpha);
    blochMxyScale = blochPrePulseMz * Math.sin(alpha);
}

function appendBlochVectorPath(Mx, My, Mz) {
    blochVectorPathX.push(Mx);
    blochVectorPathY.push(My);
    blochVectorPathZ.push(Mz);
}

function restyleBlochVector(Mx, My, Mz) {
    Plotly.restyle('bloch-vector-plot', {
        x: [[0, Mx]],
        y: [[0, My]],
        z: [[0, Mz]]
    }, [blochVectorTraceIndex]);
    Plotly.restyle('bloch-vector-plot', {
        x: [blochVectorPathX],
        y: [blochVectorPathY],
        z: [blochVectorPathZ]
    }, [blochVectorPathIndex]);
}

function initBlochVectorPlot() {
    if (typeof Plotly === 'undefined') return;
    const plotDiv = document.getElementById('bloch-vector-plot');
    if (!plotDiv) return;

    const { R2, R1, delta } = getBlochVectorParams();
    resetBlochPulseState();
    const { Mx, My, Mz } = computeBlochVector(0, R2, R1, delta);

    const axisX = {
        x: [0, 1],
        y: [0, 0],
        z: [0, 0],
        type: 'scatter3d',
        mode: 'lines',
        line: { color: '#999', width: 3 },
        hoverinfo: 'skip',
        showlegend: false
    };
    const axisY = {
        x: [0, 0],
        y: [0, 1],
        z: [0, 0],
        type: 'scatter3d',
        mode: 'lines',
        line: { color: '#999', width: 3 },
        hoverinfo: 'skip',
        showlegend: false
    };
    const axisZ = {
        x: [0, 0],
        y: [0, 0],
        z: [0, 1],
        type: 'scatter3d',
        mode: 'lines',
        line: { color: '#999', width: 3 },
        hoverinfo: 'skip',
        showlegend: false
    };
    const vectorTrace = {
        x: [0, Mx],
        y: [0, My],
        z: [0, Mz],
        type: 'scatter3d',
        mode: 'lines+markers',
        line: { color: '#7B2D3B', width: 6 },
        marker: { size: 4, color: '#7B2D3B' },
        name: 'M'
    };
    const pathTrace = {
        x: [Mx],
        y: [My],
        z: [Mz],
        type: 'scatter3d',
        mode: 'lines',
        line: { color: 'rgba(220, 53, 69, 0.7)', width: 3 },
        hoverinfo: 'skip',
        showlegend: false
    };

    const layout = {
        margin: { l: 0, r: 0, t: 10, b: 0 },
        height: 400,
        showlegend: false,
        scene: {
            aspectmode: 'cube',
            dragmode: 'orbit',
            xaxis: { range: [-1.1, 1.1], title: 'Mx', zeroline: true },
            yaxis: { range: [-1.1, 1.1], title: 'My', zeroline: true },
            zaxis: { range: [-1.1, 1.1], title: 'Mz', zeroline: true }
        }
    };

    Plotly.react('bloch-vector-plot', [axisX, axisY, axisZ, vectorTrace, pathTrace], layout, { responsive: true, displayModeBar: false });
    if (!plotDiv.dataset.interactionBound) {
        plotDiv.addEventListener('mousedown', stopBlochVectorAnimation);
        plotDiv.addEventListener('touchstart', stopBlochVectorAnimation, { passive: true });
        plotDiv.addEventListener('mouseup', startBlochVectorAnimation);
        plotDiv.addEventListener('mouseleave', startBlochVectorAnimation);
        plotDiv.addEventListener('touchend', startBlochVectorAnimation);
        plotDiv.dataset.interactionBound = 'true';
    }
    blochVectorInitialized = true;
}

function updateBlochVectorPlot(t) {
    if (!blochVectorInitialized || typeof Plotly === 'undefined') return;
    const plotDiv = document.getElementById('bloch-vector-plot');
    if (!plotDiv) return;

    const { R2, R1, delta } = getBlochVectorParams();
    const { Mx, My, Mz } = computeBlochVector(t, R2, R1, delta);

    appendBlochVectorPath(Mx, My, Mz);
    restyleBlochVector(Mx, My, Mz);
}

function startBlochVectorAnimation() {
    if (!blochVectorInitialized) initBlochVectorPlot();
    if (blochVectorAnimationId) return;
    blochVectorLastTs = null;
    blochVectorAnimationId = requestAnimationFrame(stepBlochVectorAnimation);
}

function stopBlochVectorAnimation() {
    if (blochVectorAnimationId) {
        cancelAnimationFrame(blochVectorAnimationId);
        blochVectorAnimationId = null;
    }
}

function resetBlochVectorAnimation() {
    blochVectorTime = 0;
    blochVectorPathX = [];
    blochVectorPathY = [];
    blochVectorPathZ = [];
    blochPrePulseMz = 1;
    resetBlochPulseState();
    updateBlochVectorPlot(blochVectorTime);
}

function stepBlochVectorAnimation(timestamp) {
    if (!blochVectorAnimationId) return;
    const blochTab = document.getElementById('tab-bloch');
    if (!blochTab || !blochTab.classList.contains('active')) {
        stopBlochVectorAnimation();
        return;
    }
    if (!blochVectorLastTs) {
        blochVectorLastTs = timestamp;
    }
    const dt = (timestamp - blochVectorLastTs) / 1000;
    blochVectorLastTs = timestamp;

    const { R2, R1, delta, speed, d1, d1Enabled } = getBlochVectorParams();
    const tMax = d1Enabled ? d1 : (Math.log(20) / R1);
    const effectiveDt = dt * speed;
    let remaining = effectiveDt;
    let last = null;
    while (remaining > 0) {
        const step = Math.min(blochVectorMaxStep, remaining);
        const nextTime = blochVectorTime + step;
        if (nextTime >= tMax) {
            if (d1Enabled) {
                blochPrePulseMz = computeMzAtTime(tMax, R1);
            } else {
                blochPrePulseMz = 1;
            }
            resetBlochPulseState();
            blochVectorTime = 0;
            blochVectorPathX = [];
            blochVectorPathY = [];
            blochVectorPathZ = [];
            remaining -= step;
            const overflow = nextTime - tMax;
            remaining += overflow;
            continue;
        }
        remaining -= step;
        blochVectorTime = nextTime;
        last = computeBlochVector(blochVectorTime, R2, R1, delta);
        appendBlochVectorPath(last.Mx, last.My, last.Mz);
    }
    if (last) restyleBlochVector(last.Mx, last.My, last.Mz);
    blochVectorAnimationId = requestAnimationFrame(stepBlochVectorAnimation);
}

function updateBlochPlots() {
    if (typeof Plotly === 'undefined') return;

    try {
        const d1EnabledEl = document.getElementById('bloch-d1-enabled');
        const d1Slider = document.getElementById('bloch-d1');
        const d1Enabled = d1EnabledEl ? d1EnabledEl.checked : false;
        const R2 = Math.pow(10, parseFloat(document.getElementById('bloch-R2').value));
        const R1 = Math.pow(10, parseFloat(document.getElementById('bloch-R1').value));
        const delta = parseFloat(document.getElementById('bloch-delta').value);
        const flipAngle = parseFloat(document.getElementById('bloch-flip').value);
        const ftMin = parseFloat(document.getElementById('bloch-ftMin').value);
        const ftMax = parseFloat(document.getElementById('bloch-ftMax').value);
        const d1 = d1Slider ? parseFloat(d1Slider.value) : 1.0;

        // Update slider value displays
        document.getElementById('bloch-R2-value').textContent = R2.toFixed(1);
        document.getElementById('bloch-R1-value').textContent = R1.toFixed(1);
        document.getElementById('bloch-delta-value').textContent = delta.toFixed(0);
        document.getElementById('bloch-flip-value').textContent = flipAngle.toFixed(0);
        document.getElementById('bloch-ftMin-value').textContent = ftMin.toFixed(1);
        document.getElementById('bloch-ftMax-value').textContent = ftMax.toFixed(1);
        const speed = Math.pow(10, parseFloat(document.getElementById('bloch-speed').value));
        document.getElementById('bloch-speed-value').textContent = speed.toFixed(4);
        if (document.getElementById('bloch-d1-value')) {
            document.getElementById('bloch-d1-value').textContent = d1.toFixed(2);
        }
        if (document.getElementById('bloch-ernst-value')) {
            const E1 = Math.exp(-R1 * d1);
            const ernstRad = Math.acos(Math.max(-1, Math.min(1, E1)));
            const ernstDeg = ernstRad * 180 / Math.PI;
            document.getElementById('bloch-ernst-value').textContent = ernstDeg.toFixed(1);
        }
        if (d1Slider) {
            d1Slider.disabled = !d1Enabled;
            d1Slider.classList.toggle('disabled', !d1Enabled);
        }

        // Derived quantities
        const FWHM = R2 / Math.PI;

        // Simulate single-spin FID
        const { timePoints, fid } = simulateBlochFID(delta, R2, flipAngle);
        const { freqAxis, spectrum } = computeSpectrum(fid, timePoints);

        const spectrumReal = spectrum.map(c => c.real);
        const fidReal = fid.map(c => c.real);
        const fidImag = fid.map(c => c.imag);

        // ---- Spectrum plot ----
        const spectrumTrace = {
            x: freqAxis,
            y: spectrumReal,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#7B2D3B', width: 2 },
            name: 'Spectrum'
        };

        const ruleLine = {
            x: [delta, delta],
            y: [ftMin, ftMax],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#A3445A', dash: 'dash', width: 2 },
            name: `ΔΩ = ${delta} Hz`,
            showlegend: true
        };

        const spectrumLayout = {
            title: `NMR Spectrum — Single Spin<br>FWHM = ${FWHM.toFixed(1)} Hz (R<sub>2</sub> = ${R2.toFixed(1)} s<sup>-1</sup>)`,
            xaxis: { title: 'Frequency (Hz)', autorange: true },
            yaxis: { title: 'Intensity', range: [ftMin, ftMax] },
            height: 400,
            margin: { l: 60, r: 20, t: 60, b: 50 },
            hovermode: 'closest',
            showlegend: true
        };

        Plotly.react('bloch-spectrum-plot', [spectrumTrace, ruleLine], spectrumLayout, { responsive: true, displayModeBar: true });

        // ---- FID plot ----
        const fidRealTrace = {
            x: timePoints,
            y: fidReal,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#7B2D3B', width: 2 },
            name: 'Real (M<sub>x</sub>)'
        };

        const fidImagTrace = {
            x: timePoints,
            y: fidImag,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#E0829B', width: 2 },
            name: 'Imaginary (M<sub>y</sub>)'
        };

        const fidLayout = {
            title: `Free Induction Decay — α = ${flipAngle}&deg;, ΔΩ = ${delta} Hz`,
            xaxis: { title: 'Time (s)', autorange: true },
            yaxis: { title: 'M<sub>x</sub> + iM<sub>y</sub>', autorange: true },
            height: 400,
            margin: { l: 60, r: 20, t: 40, b: 50 },
            hovermode: 'closest',
            showlegend: true
        };

        Plotly.react('bloch-fid-plot', [fidRealTrace, fidImagTrace], fidLayout, { responsive: true, displayModeBar: true });
        resetBlochVectorAnimation();
    } catch (error) {
        console.error('Error updating Bloch plots:', error);
    }
}

function initBloch() {
    if (typeof Plotly === 'undefined') {
        setTimeout(initBloch, 100);
        return;
    }
    blochInitialized = true;

    const ids = ['bloch-R2', 'bloch-R1', 'bloch-delta', 'bloch-flip',
                 'bloch-speed', 'bloch-d1', 'bloch-d1-enabled',
                 'bloch-ftMin', 'bloch-ftMax'];
    ids.forEach(id => document.getElementById(id).addEventListener('input', updateBlochPlots));

    updateBlochPlots();
    initBlochVectorPlot();
    startBlochVectorAnimation();
}

// ============================================================
// Tab switching
// ============================================================
let simulationInitialized = false;

function initTabs() {
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Deactivate all
            buttons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            // Activate clicked
            btn.classList.add('active');
            const tabId = 'tab-' + btn.dataset.tab;
            document.getElementById(tabId).classList.add('active');

            // Lazy-init / resize each tab's plots
            if (btn.dataset.tab === 'bloch') {
                Plotly.Plots.resize('bloch-spectrum-plot');
                Plotly.Plots.resize('bloch-fid-plot');
                Plotly.Plots.resize('bloch-vector-plot');
                startBlochVectorAnimation();
            }

            if (btn.dataset.tab === 'simulation' && !simulationInitialized) {
                initSimulation();
            }
            if (btn.dataset.tab === 'simulation') {
                Plotly.Plots.resize('spectrum-plot');
                Plotly.Plots.resize('fid-plot');
            }

            if (btn.dataset.tab !== 'bloch') {
                stopBlochVectorAnimation();
            }

            if (btn.dataset.tab === 'animation' && !animationInitialized) {
                initAnimation();
            }
            if (btn.dataset.tab === 'animation') {
                Plotly.Plots.resize('animation-plot');
            }

            if (btn.dataset.tab === 'dispersion' && !dispersionInitialized) {
                initDispersion();
            }
            if (btn.dataset.tab === 'dispersion') {
                Plotly.Plots.resize('dispersion-plot');
                Plotly.Plots.resize('dispersion-spectrum-plot');
            }
        });
    });
}

// ============================================================
// Precession animation (dots on concentric circles)
// ============================================================
let animationInitialized = false;
let animRunning = false;
let animFrameId = null;
let animElapsed = 0;         // total elapsed animation time in seconds
let animLastTimestamp = 0;   // last frame timestamp
let precessionDirection = 1; // +1 normal, -1 after echo
let viewMode = 'concentric'; // 'concentric' or 'vector'
let singleEchoTau = null;    // when in single-echo mode, τ in seconds (from slider at Play, or from Echo button press)

// Histogram bins for y-component distribution
const HIST_NBINS = 40;
const HIST_BIN_EDGES = [];
const HIST_BIN_CENTERS = [];
for (let i = 0; i <= HIST_NBINS; i++) {
    HIST_BIN_EDGES.push(-1 + (2 * i / HIST_NBINS));
}
for (let i = 0; i < HIST_NBINS; i++) {
    HIST_BIN_CENTERS.push((HIST_BIN_EDGES[i] + HIST_BIN_EDGES[i + 1]) / 2);
}

// Mx(t) time series for bottom plot
let mxTimes = [];
let mxValues = [];

// Color for rate A spins and rate B spins
const COLOR_A = '#7B2D3B';
const COLOR_B = '#D4764E';

// Per-spin state
let spinAssignment = []; // 0 = rate A, 1 = rate B
let spinPhase = [];      // accumulated phase angle (radians) per spin

// Build a random 50/50 assignment for n spins
function randomizeAssignment(n) {
    const half = Math.floor(n / 2);
    spinAssignment = [];
    for (let i = 0; i < n; i++) {
        spinAssignment.push(i < half ? 0 : 1);
    }
    // Fisher-Yates shuffle
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [spinAssignment[i], spinAssignment[j]] = [spinAssignment[j], spinAssignment[i]];
    }
    // Reset phases to 0
    spinPhase = new Array(n).fill(0);
}

// Apply stochastic exchange: each spin has probability k_ex * dt of swapping
// its assignment (A↔B) while keeping its current phase
function applyExchange(dt) {
    const kEx = parseFloat(document.getElementById('anim-exchange').value);
    if (kEx <= 0) return;
    const pSwap = kEx * dt; // probability of exchange in this time step
    const n = spinAssignment.length;
    for (let i = 0; i < n; i++) {
        if (Math.random() < pSwap) {
            spinAssignment[i] = spinAssignment[i] === 0 ? 1 : 0;
        }
    }
}

function initAnimation() {
    animationInitialized = true;

    const rateSlider = document.getElementById('anim-rate');
    const rate2Slider = document.getElementById('anim-rate2');
    const exchangeSlider = document.getElementById('anim-exchange');
    const echoRateSlider = document.getElementById('anim-echo-rate');
    const numEchoesSlider = document.getElementById('anim-num-echoes');
    const singleEchoCheckbox = document.getElementById('anim-single-echo');
    const singleEchoControls = document.getElementById('anim-single-echo-controls');
    const tauSlider = document.getElementById('anim-tau');
    const cpmgLockCheckbox = document.getElementById('anim-cpmg-lock');
    const circlesSlider = document.getElementById('anim-circles');
    const playBtn = document.getElementById('anim-play');

    const CPMG_FIXED_TIME = 10; // seconds

    // When CPMG Dispersion is on, auto-compute number of echoes from ν_CPMG
    function syncEchoesFromRate() {
        if (!cpmgLockCheckbox.checked) return;
        const nu = parseFloat(echoRateSlider.value);
        if (nu <= 0) return;
        let N = Math.round(CPMG_FIXED_TIME * nu);
        if (N < 2) N = 2;
        if (N % 2 !== 0) N += 1;
        const max = parseInt(numEchoesSlider.max);
        if (N > max) N = max;
        numEchoesSlider.value = N;
        document.getElementById('anim-num-echoes-value').textContent = N;
    }

    // When Single Echo is on, show τ slider and disable ν_CPMG / num echoes (they’re not used)
    function applySingleEcho(enabled) {
        singleEchoControls.style.display = enabled ? '' : 'none';
        echoRateSlider.disabled = enabled;
        numEchoesSlider.disabled = enabled;
        if (!enabled) singleEchoTau = null;
    }

    function resetAnim() {
        animElapsed = 0;
        precessionDirection = 1;
        spinPhase = new Array(spinAssignment.length).fill(0);
        drawAnimFrame();
    }

    // Display value updates
    rateSlider.addEventListener('input', () => {
        document.getElementById('anim-rate-value').textContent =
            parseFloat(rateSlider.value).toFixed(2);
        resetAnim();
    });
    rate2Slider.addEventListener('input', () => {
        document.getElementById('anim-rate2-value').textContent =
            parseFloat(rate2Slider.value).toFixed(2);
        resetAnim();
    });
    exchangeSlider.addEventListener('input', () => {
        document.getElementById('anim-exchange-value').textContent =
            parseFloat(exchangeSlider.value).toFixed(1);
    });
    echoRateSlider.addEventListener('input', () => {
        document.getElementById('anim-echo-rate-value').textContent =
            parseFloat(echoRateSlider.value).toFixed(1);
        syncEchoesFromRate();
    });
    numEchoesSlider.addEventListener('input', () => {
        document.getElementById('anim-num-echoes-value').textContent =
            numEchoesSlider.value;
    });
    tauSlider.addEventListener('input', () => {
        document.getElementById('anim-tau-value').textContent =
            parseFloat(tauSlider.value).toFixed(1);
    });

    singleEchoCheckbox.addEventListener('change', () => {
        if (singleEchoCheckbox.checked) {
            cpmgLockCheckbox.checked = false;
        }
        applySingleEcho(singleEchoCheckbox.checked);
        numEchoesSlider.disabled = singleEchoCheckbox.checked || cpmgLockCheckbox.checked;
    });

    cpmgLockCheckbox.addEventListener('change', () => {
        if (cpmgLockCheckbox.checked) {
            singleEchoCheckbox.checked = false;
            applySingleEcho(false);
        }
        echoRateSlider.disabled = false;
        numEchoesSlider.disabled = cpmgLockCheckbox.checked;
        syncEchoesFromRate();
    });
    circlesSlider.addEventListener('input', () => {
        document.getElementById('anim-circles-value').textContent = circlesSlider.value;
        randomizeAssignment(parseInt(circlesSlider.value));
        buildAnimPlot();
        resetAnim();
    });

    playBtn.addEventListener('click', () => {
        animRunning = !animRunning;
        playBtn.textContent = animRunning ? 'Pause' : 'Play';
        if (animRunning) {
            animElapsed = 0;
            precessionDirection = 1;
            spinPhase = new Array(spinAssignment.length).fill(0);
            if (singleEchoCheckbox.checked) {
                singleEchoTau = parseFloat(tauSlider.value);
            } else {
                singleEchoTau = null;
            }
            buildAnimPlot();
            initSignalPlot();
            initMxPlot();
            drawAnimFrame();
            updateSignalPlot();
            updateMxPlot();
            animLastTimestamp = performance.now();
            animateLoop();
        } else {
            singleEchoTau = null;
        }
    });

    // View mode toggle
    const viewConcentricBtn = document.getElementById('view-concentric');
    const viewVectorBtn = document.getElementById('view-vector');

    viewConcentricBtn.addEventListener('click', () => {
        viewMode = 'concentric';
        viewConcentricBtn.classList.add('active');
        viewVectorBtn.classList.remove('active');
        buildAnimPlot();
        drawAnimFrame();
    });
    viewVectorBtn.addEventListener('click', () => {
        viewMode = 'vector';
        viewVectorBtn.classList.add('active');
        viewConcentricBtn.classList.remove('active');
        buildAnimPlot();
        drawAnimFrame();
    });

    // Initial assignment
    const nCircles = parseInt(circlesSlider.value);
    randomizeAssignment(nCircles);

    // Draw concentric circle outlines and initial dot positions
    buildAnimPlot();
    initSignalPlot();
    initMxPlot();
    drawAnimFrame(); // draw the stopped frame (all dots on +x axis)
}

function buildAnimPlot() {
    const nCircles = parseInt(document.getElementById('anim-circles').value);

    const layout = {
        xaxis: {
            range: [-1.25, 1.25],
            scaleanchor: 'y',
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        yaxis: {
            range: [-1.25, 1.25],
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        shapes: [],
        margin: { l: 10, r: 10, t: 10, b: 10 },
        plot_bgcolor: '#fafafa',
        height: 500,
        showlegend: false
    };

    if (viewMode === 'concentric') {
        // Concentric circles as background shapes
        const shapes = [];
        for (let c = 0; c < nCircles; c++) {
            const r = (c + 1) / nCircles;
            shapes.push({
                type: 'circle',
                xref: 'x', yref: 'y',
                x0: -r, y0: -r, x1: r, y1: r,
                line: { color: 'rgba(150,150,150,0.3)', width: 1 },
                fillcolor: 'transparent'
            });
        }
        layout.shapes = shapes;

        Plotly.react('animation-plot', [{
            x: [], y: [],
            mode: 'markers',
            marker: { size: 5, color: [] },
            type: 'scatter'
        }], layout, { responsive: true, displayModeBar: false });
    } else {
        // Vector mode: one outer circle + line traces (one per spin)
        layout.shapes = [{
            type: 'circle',
            xref: 'x', yref: 'y',
            x0: -1, y0: -1, x1: 1, y1: 1,
            line: { color: 'rgba(150,150,150,0.4)', width: 2 },
            fillcolor: 'transparent'
        }];

        // Create one trace per spin (line from center to dot)
        const traces = [];
        for (let c = 0; c < nCircles; c++) {
            const color = spinAssignment[c] === 0 ? COLOR_A : COLOR_B;
            traces.push({
                x: [0, 1], y: [0, 0],
                mode: 'lines+markers',
                line: { color: color, width: 1.5 },
                marker: { size: 5, color: color, symbol: 'circle' },
                type: 'scatter'
            });
        }

        Plotly.react('animation-plot', traces, layout, { responsive: true, displayModeBar: false });
    }
}

// Advance all spin phases by dt seconds
function stepPhases(dt) {
    const rateA = parseFloat(document.getElementById('anim-rate').value);
    const rateB = parseFloat(document.getElementById('anim-rate2').value);
    const n = spinAssignment.length;

    for (let c = 0; c < n; c++) {
        const rate = spinAssignment[c] === 0 ? rateA : rateB;
        spinPhase[c] += 2 * Math.PI * rate * dt * precessionDirection;
    }
}

function initSignalPlot() {
    const n = spinAssignment.length || 100;
    const maxCount = Math.ceil(n * 0.6);
    const layout = {
        xaxis: { range: [-maxCount, maxCount], zeroline: true,
                 zerolinecolor: 'rgba(100,100,100,0.6)', zerolinewidth: 1, showticklabels: false },
        yaxis: { range: [-1.25, 1.25], showgrid: false, zeroline: true,
                 zerolinecolor: 'rgba(150,150,150,0.5)', showticklabels: false },
        margin: { l: 10, r: 10, t: 10, b: 10 },
        height: 500,
        plot_bgcolor: '#fafafa',
        bargap: 0.05,
        showlegend: false
    };
    Plotly.react('signal-plot', [
        {
            x: new Array(HIST_NBINS).fill(0),
            y: HIST_BIN_CENTERS,
            type: 'bar',
            orientation: 'h',
            marker: { color: 'black', opacity: 0.8 },
            width: 2 / HIST_NBINS * 0.9
        },
        {
            x: new Array(HIST_NBINS).fill(0),
            y: HIST_BIN_CENTERS,
            type: 'bar',
            orientation: 'h',
            marker: { color: 'black', opacity: 0.8 },
            width: 2 / HIST_NBINS * 0.9
        }
    ], layout, { responsive: true, displayModeBar: false });
}

function updateSignalPlot() {
    const n = spinPhase.length;
    if (n === 0) return;

    const binWidth = 2 / HIST_NBINS;
    const countsPositive = new Array(HIST_NBINS).fill(0);
    const countsNegative = new Array(HIST_NBINS).fill(0);

    for (let i = 0; i < n; i++) {
        const phase = spinPhase[i] || 0;
        const xVal = Math.cos(phase);
        const yVal = Math.sin(phase);
        let bin = Math.floor((yVal + 1) / binWidth);
        if (bin < 0) bin = 0;
        if (bin >= HIST_NBINS) bin = HIST_NBINS - 1;
        if (xVal >= 0) {
            countsPositive[bin]++;
        } else {
            countsNegative[bin]++;
        }
    }

    const maxCount = Math.max(
        Math.max(...countsPositive),
        Math.max(...countsNegative),
        n * 0.3
    );
    const symRange = maxCount * 1.15;

    Plotly.update('signal-plot',
        {
            x: [countsPositive, countsNegative.map(c => -c)]
        },
        {
            'xaxis.range': [-symRange, symRange]
        }
    );
}

function computeMx() {
    const n = spinPhase.length;
    if (n === 0) return 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += Math.cos(spinPhase[i] || 0);
    }
    return sum / n;
}

function initMxPlot() {
    mxTimes = [];
    mxValues = [];
    const totalTime = getTotalTime();
    const xMax = (totalTime !== Infinity && totalTime > 0) ? totalTime : 10;
    const layout = {
        xaxis: { title: 'Time (s)', range: [0, xMax], zeroline: false },
        yaxis: { title: 'M<sub>x</sub> / M<sub>0</sub>', range: [-1.1, 1.1], zeroline: true, zerolinecolor: '#ccc' },
        margin: { l: 55, r: 20, t: 8, b: 40 },
        height: 220,
        plot_bgcolor: '#fafafa'
    };
    Plotly.react('mx-plot', [{
        x: [],
        y: [],
        mode: 'lines',
        line: { color: '#333', width: 2 },
        type: 'scatter'
    }], layout, { responsive: true, displayModeBar: false });
}

function updateMxPlot() {
    mxTimes.push(animElapsed);
    mxValues.push(computeMx());
    const totalTime = getTotalTime();
    const xMax = (totalTime !== Infinity && totalTime > 0) ? Math.max(totalTime, animElapsed * 1.05) : Math.max(10, animElapsed * 1.05);
    Plotly.update('mx-plot',
        { x: [mxTimes], y: [mxValues] },
        { 'xaxis.range': [0, xMax] }
    );
}

function drawAnimFrame() {
    const nCircles = parseInt(document.getElementById('anim-circles').value);

    if (viewMode === 'concentric') {
        const xs = [];
        const ys = [];
        const colors = [];

        for (let c = 0; c < nCircles; c++) {
            const r = (c + 1) / nCircles;
            const angle = spinPhase[c] || 0;
            xs.push(r * Math.cos(angle));
            ys.push(r * Math.sin(angle));
            colors.push(spinAssignment[c] === 0 ? COLOR_A : COLOR_B);
        }

        // Rebuild shapes in case nCircles changed
        const shapes = [];
        for (let c = 0; c < nCircles; c++) {
            const r = (c + 1) / nCircles;
            shapes.push({
                type: 'circle',
                xref: 'x', yref: 'y',
                x0: -r, y0: -r, x1: r, y1: r,
                line: { color: 'rgba(150,150,150,0.3)', width: 1 },
                fillcolor: 'transparent'
            });
        }

        Plotly.update('animation-plot',
            { x: [xs], y: [ys], 'marker.color': [colors] },
            { shapes: shapes }
        );
    } else {
        // Vector mode: update each trace (one per spin)
        // Guard: if trace count doesn't match, rebuild the plot
        const plotDiv = document.getElementById('animation-plot');
        if (!plotDiv.data || plotDiv.data.length !== nCircles) {
            buildAnimPlot();
        }

        const xArrays = [];
        const yArrays = [];
        const colorArrays = [];

        for (let c = 0; c < nCircles; c++) {
            const angle = spinPhase[c] || 0;
            const tipX = Math.cos(angle);
            const tipY = Math.sin(angle);
            const color = spinAssignment[c] === 0 ? COLOR_A : COLOR_B;
            xArrays.push([0, tipX]);
            yArrays.push([0, tipY]);
            colorArrays.push(color);
        }

        // Update all traces at once
        const traceIndices = Array.from({ length: nCircles }, (_, i) => i);
        Plotly.update('animation-plot',
            {
                x: xArrays,
                y: yArrays,
                'line.color': colorArrays,
                'marker.color': colorArrays
            },
            {},
            traceIndices
        );
    }
}

// CPMG sequence: τ - 2τ - 2τ - ... - 2τ - τ
// N refocusing pulses at times: τ, 3τ, 5τ, ..., (2N-1)τ
// Total time: 2Nτ
// Segments:
//   [0, τ)       → direction +1  (first half-echo)
//   [τ, 3τ)      → direction -1  (full echo)
//   [3τ, 5τ)     → direction +1  (full echo)
//   ...
//   [(2N-1)τ, 2Nτ] → last half-echo
//
function getEchoTau() {
    const singleEchoEl = document.getElementById('anim-single-echo');
    const cpmgEl = document.getElementById('anim-cpmg-lock');
    if (singleEchoEl && singleEchoEl.checked && singleEchoTau != null) {
        return singleEchoTau;
    }
    if (!cpmgEl || !cpmgEl.checked) {
        return Infinity; // free precession when neither single echo nor CPMG is on
    }
    const echoRate = parseFloat(document.getElementById('anim-echo-rate').value);
    if (echoRate <= 0) return Infinity;
    return 1 / (2 * echoRate);
}

function getDirection(t) {
    const tau = getEchoTau();
    if (tau === Infinity) return 1; // no echoes, free precession

    if (t < tau) {
        return 1; // first segment: duration τ, direction +1
    }
    // After the first τ, segments are 2τ long
    const tPastFirst = t - tau;
    const segIndex = 1 + Math.floor(tPastFirst / (2 * tau));
    return (segIndex % 2 === 0) ? 1 : -1;
}

// Find the time of the next direction boundary after time t
function getNextBoundary(t) {
    const tau = getEchoTau();
    if (tau === Infinity) return Infinity;

    // Boundaries are at: τ, 3τ, 5τ, 7τ, ...
    if (t < tau) return tau;
    const tPastFirst = t - tau;
    const segIndex = Math.floor(tPastFirst / (2 * tau));
    return tau + (segIndex + 1) * 2 * tau;
}

// Total duration: single-echo mode 4τ, CPMG 2*N*τ, else Infinity (free precession)
function getTotalTime() {
    const singleEchoEl = document.getElementById('anim-single-echo');
    const cpmgEl = document.getElementById('anim-cpmg-lock');
    if (singleEchoEl && singleEchoEl.checked && singleEchoTau != null) {
        return 4 * singleEchoTau;
    }
    if (!cpmgEl || !cpmgEl.checked) {
        return Infinity; // free precession
    }
    const numEchoes = parseInt(document.getElementById('anim-num-echoes').value);
    const tau = getEchoTau();
    if (tau === Infinity) return Infinity;
    return 2 * numEchoes * tau;
}

function animateLoop() {
    if (!animRunning) return;

    const now = performance.now();
    let dt = (now - animLastTimestamp) / 1000;
    animLastTimestamp = now;

    // Cap dt to avoid spiral of death if browser lags
    if (dt > 0.1) dt = 0.1;

    const totalTime = getTotalTime();

    // Clamp if we'd overshoot the total experiment time
    if (animElapsed + dt >= totalTime) {
        dt = totalTime - animElapsed;
    }

    // Process the time step, splitting at direction boundaries
    // so the direction is always correct for each sub-step.
    // Apply exchange once for the whole frame (not per sub-step)
    // to avoid excessive random number generation.
    let remaining = dt;
    let iterations = 0;
    const maxIterations = 200;

    while (remaining > 1e-10 && iterations < maxIterations) {
        precessionDirection = getDirection(animElapsed);

        // How far to the next direction flip?
        const nextBoundary = getNextBoundary(animElapsed);
        let stepToTake = nextBoundary - animElapsed;

        // Guard against floating point issues
        if (stepToTake < 1e-12) {
            animElapsed += 1e-12;
            continue;
        }

        stepToTake = Math.min(remaining, stepToTake);
        stepPhases(stepToTake);
        animElapsed += stepToTake;
        remaining -= stepToTake;
        iterations++;
    }

    // Apply exchange once per frame using total dt
    applyExchange(dt);

    drawAnimFrame();
    updateSignalPlot();
    updateMxPlot();

    // Stop if we've reached the end of the echo train
    if (animElapsed >= totalTime) {
        animRunning = false;
        document.getElementById('anim-play').textContent = 'Play';
        singleEchoTau = null;
        return;
    }

    animFrameId = requestAnimationFrame(animateLoop);
}

// ============================================================
// CPMG Relaxation Dispersion simulation
// ============================================================

let dispersionInitialized = false;

// Standard experimental νCPMG values (Hz)
const DISP_NU_CPMG = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 6000];
const DISP_NU_ECPMG = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 15000, 20000, 25000, 30000];

// Pre-generated Gaussian noise arrays (one per field)
let dispNoise1 = [];
let dispNoise2 = [];

function generateGaussianNoise(n) {
    const noise = [];
    for (let i = 0; i < n; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        noise.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
    return noise;
}

function generateDispNoise() {
    // Generate enough noise for the larger array
    const maxLen = Math.max(DISP_NU_CPMG.length, DISP_NU_ECPMG.length);
    dispNoise1 = generateGaussianNoise(maxLen);
    dispNoise2 = generateGaussianNoise(maxLen);
}

// Fast-exchange limit / Luz-Meiboom equation (Eq. 2 in Farber & Mittermaier 2015)
// R2 = pA*R2A0 + pB*R2B0 + (Φex/kex)*(1 - tanh(kex/(4νCPMG)) / (kex/(4νCPMG)))
// where Φex = pA*pB*Δω²
function fastLimitR2(nuCPMG, kex, pB, dw, R20A, R20B) {
    const pA = 1 - pB;
    const Phiex = pA * pB * dw * dw;
    const R20avg = pA * R20A + pB * R20B;
    if (kex <= 0) return R20avg;
    const Rex = Phiex / kex;
    if (nuCPMG <= 0) return R20avg + Rex;

    const x = kex / (4 * nuCPMG);
    return R20avg + Rex * (1 - Math.tanh(x) / x);
}

// Carver-Richards equation for two-site exchange (Eq. 3 in Farber & Mittermaier 2015)
// Full solution valid for all exchange timescales (slow, intermediate, fast)
//
// R2eff = ½(R2A0 + R2B0 + kex - 2νCPMG·acosh(D+·cosh(η+) - D-·cos(η-)))
//
// Parameters:
//   nuCPMG  – CPMG pulse repetition rate (Hz)
//   kex     – exchange rate kAB+kBA (s⁻¹)
//   pB      – minor state population (0 < pB < 1)
//   dw      – chemical shift difference ωB−ωA (rad/s)
//   R20A/B  – intrinsic transverse relaxation rates (s⁻¹)
function carverRichards(nuCPMG, kex, pB, dw, R20A, R20B) {
    const pA = 1 - pB;
    const R20avg = pA * R20A + pB * R20B;

    if (nuCPMG <= 0) return R20avg;
    if (kex <= 0 || pB <= 0 || pB >= 1 || Math.abs(dw) < 1e-6) return R20avg;

    const tcp = 1 / (2 * nuCPMG);
    const dR = R20A - R20B;
    const dk = (pB - pA) * kex;
    const dRdk = dR + dk;

    const psi = dRdk * dRdk - dw * dw + 4 * pA * pB * kex * kex;
    const xi = 2 * dw * dRdk;
    const psi2xi2 = Math.sqrt(psi * psi + xi * xi);

    // D± = ½(±1 + (ψ + 2Δω²) / √(ψ² + ξ²))  — Palmer et al. (2001) Eq. 16
    const Dplus = 0.5 * (1 + (psi + 2 * dw * dw) / psi2xi2);
    const Dminus = 0.5 * (-1 + (psi + 2 * dw * dw) / psi2xi2);

    const etaPlus = tcp / Math.SQRT2 * Math.sqrt(Math.max(0, psi + psi2xi2));
    const etaMinus = tcp / Math.SQRT2 * Math.sqrt(Math.max(0, -psi + psi2xi2));

    // Overflow protection: cosh overflows for arguments > ~710
    if (etaPlus > 700) {
        return fastLimitR2(nuCPMG, kex, pB, dw, R20A, R20B);
    }

    let acoshArg = Dplus * Math.cosh(etaPlus) - Dminus * Math.cos(etaMinus);
    if (acoshArg < 1) acoshArg = 1;

    let R2eff = 0.5 * (R20A + R20B + kex - 2 * nuCPMG * Math.acosh(acoshArg));
    if (R2eff < R20avg) R2eff = R20avg;

    return R2eff;
}

function updateDispersionPlots() {
    const kex = Math.pow(10, parseFloat(document.getElementById('disp-kex').value));
    const pB_pct = parseFloat(document.getElementById('disp-pB').value);
    const pB = pB_pct / 100;
    const pA = 1 - pB;
    const dwPpm = parseFloat(document.getElementById('disp-dw').value);

    // Larmor frequencies in MHz at 14.1 T and 18.8 T for each nucleus
    const nucleusFreqs = {
        '15N': { f600: 60.8,  f800: 81.1  },
        '13C': { f600: 150.9, f800: 201.2 },
        '1H':  { f600: 600.0, f800: 800.0 }
    };
    const nucleus = document.getElementById('disp-nucleus').value;
    const { f600, f800 } = nucleusFreqs[nucleus];

    const dwHz = dwPpm * f600;       // Δν in Hz at 14.1 T
    const dwHz800 = dwPpm * f800;    // Δν in Hz at 18.8 T
    const dw = 2 * Math.PI * dwHz;   // Δω in rad/s at 14.1 T
    const R20 = Math.pow(10, parseFloat(document.getElementById('disp-R20').value));
    const R20_800 = Math.pow(10, parseFloat(document.getElementById('disp-R20-800').value));
    const noiseSigma = parseFloat(document.getElementById('disp-noise').value);
    const showDualField = document.getElementById('disp-dual-field').checked;
    const showFast = document.getElementById('disp-show-fast').checked;
    const showEcpmg = document.getElementById('disp-ecpmg').checked;

    // Show/hide R20 (18.8 T) slider
    const r20800Group = document.getElementById('disp-R20-800-group');
    if (r20800Group) r20800Group.style.display = showDualField ? '' : 'none';

    // Select frequency range based on CPMG / eCPMG mode
    const DISP_NU_CPMG_EXP = showEcpmg ? DISP_NU_ECPMG : DISP_NU_CPMG;
    const nuMax = showEcpmg ? 30000 : 6000;

    // Update displayed values
    document.getElementById('disp-kex-value').textContent = kex.toFixed(0);
    document.getElementById('disp-pB-value').textContent = pB_pct.toFixed(1);
    document.getElementById('disp-dw-value').textContent = dwPpm.toFixed(2);
    document.getElementById('disp-dw-hz600').textContent = dwHz.toFixed(0);
    document.getElementById('disp-dw-hz800').textContent = dwHz800.toFixed(0);
    document.getElementById('disp-R20-value').textContent = R20.toFixed(1);
    document.getElementById('disp-R20-800-value').textContent = R20_800.toFixed(1);
    document.getElementById('disp-noise-value').textContent = noiseSigma.toFixed(1);

    // Exchange regime indicator (kex / |Δω|)
    const ratio = Math.abs(dw) > 0 ? kex / Math.abs(dw) : Infinity;
    let regime;
    if (ratio > 3) regime = 'Fast';
    else if (ratio > 0.5) regime = 'Intermediate';
    else regime = 'Slow';

    document.getElementById('disp-ratio-value').textContent = isFinite(ratio) ? ratio.toFixed(2) : '∞';
    document.getElementById('disp-regime-label').textContent = regime;

    const Rex = kex > 0 ? pA * pB * dw * dw / kex : 0;
    document.getElementById('disp-Rex-value').textContent = Rex.toFixed(1);

    // ---- Dispersion curve ----
    // Smooth theoretical curves (200 points from 25 to nuMax Hz)
    const nuSmooth = [];
    for (let i = 0; i < 200; i++) {
        nuSmooth.push(25 + (nuMax - 25) * i / 199);
    }
    const R2CR = nuSmooth.map(nu => carverRichards(nu, kex, pB, dw, R20, R20));
    const R2Fast = nuSmooth.map(nu => fastLimitR2(nu, kex, pB, dw, R20, R20));

    // Simulated experimental data at 14.1 T (600 MHz ¹H / ~60.8 MHz ¹⁵N)
    const dataR2_600 = DISP_NU_CPMG_EXP.map((nu, i) =>
        carverRichards(nu, kex, pB, dw, R20, R20) + noiseSigma * (dispNoise1[i] || 0)
    );

    const traces = [];

    // Exact Carver-Richards curve (14.1 T)
    traces.push({
        x: nuSmooth, y: R2CR,
        mode: 'lines',
        line: { color: '#7B2D3B', width: 3 },
        name: 'Carver-Richards (14.1 T)',
        type: 'scatter'
    });

    // Simulated data points (14.1 T)
    traces.push({
        x: DISP_NU_CPMG_EXP, y: dataR2_600,
        mode: 'markers',
        marker: { color: '#7B2D3B', size: 8, symbol: 'circle', line: { color: '#5C1A2A', width: 1 } },
        error_y: { type: 'constant', value: noiseSigma, visible: noiseSigma > 0, color: '#7B2D3B', thickness: 1.5 },
        name: 'Data (14.1 T)',
        type: 'scatter'
    });

    // Fast-limit (Luz-Meiboom) approximation
    if (showFast) {
        traces.push({
            x: nuSmooth, y: R2Fast,
            mode: 'lines',
            line: { color: '#C4956A', width: 2, dash: 'dash' },
            name: 'Luz-Meiboom (LM) approx.',
            type: 'scatter'
        });
    }

    // Second magnetic field: 18.8 T (800 MHz ¹H / ~81.1 MHz ¹⁵N)
    // Δω derived from ppm × ¹⁵N Larmor frequency at 18.8 T
    if (showDualField) {
        const dw800 = 2 * Math.PI * dwHz800;

        const R2CR800 = nuSmooth.map(nu => carverRichards(nu, kex, pB, dw800, R20_800, R20_800));
        const dataR2_800 = DISP_NU_CPMG_EXP.map((nu, i) =>
            carverRichards(nu, kex, pB, dw800, R20_800, R20_800) + noiseSigma * (dispNoise2[i] || 0)
        );

        traces.push({
            x: nuSmooth, y: R2CR800,
            mode: 'lines',
            line: { color: '#D4764E', width: 3 },
            name: 'Carver-Richards (18.8 T)',
            type: 'scatter'
        });
        traces.push({
            x: DISP_NU_CPMG_EXP, y: dataR2_800,
            mode: 'markers',
            marker: { color: '#D4764E', size: 8, symbol: 'diamond', line: { color: '#B5613A', width: 1 } },
            error_y: { type: 'constant', value: noiseSigma, visible: noiseSigma > 0, color: '#D4764E', thickness: 1.5 },
            name: 'Data (18.8 T)',
            type: 'scatter'
        });
    }

    // R₂⁰ baseline (dotted) — 14.1 T
    traces.push({
        x: [0, nuMax + 50], y: [R20, R20],
        mode: 'lines',
        line: { color: '#7B2D3B', width: 1, dash: 'dot' },
        name: 'R₂⁰ (14.1 T)',
        showlegend: true,
        type: 'scatter'
    });

    // R₂⁰ baseline (dotted) — 18.8 T
    if (showDualField) {
        traces.push({
            x: [0, nuMax + 50], y: [R20_800, R20_800],
            mode: 'lines',
            line: { color: '#D4764E', width: 1, dash: 'dot' },
            name: 'R₂⁰ (18.8 T)',
            showlegend: true,
            type: 'scatter'
        });
    }

    const dispLayout = {
        xaxis: { title: 'ν<sub>CPMG</sub> (Hz)', range: [0, nuMax + 50], zeroline: false },
        yaxis: { title: 'R<sub>2,eff</sub> (s⁻¹)', zeroline: false },
        height: 400,
        margin: { l: 65, r: 20, t: 20, b: 55 },
        hovermode: 'closest',
        showlegend: true,
        legend: { x: 0.55, y: 0.98 }
    };

    Plotly.react('dispersion-plot', traces, dispLayout, { responsive: true, displayModeBar: true });

    // ---- NMR Spectrum (exchange-broadened lineshape) ----
    // Map dispersion parameters to Bloch-McConnell simulation
    const kAB = pB * kex;
    const kBA = pA * kex;
    const deltaA = -dwHz / 2;
    const deltaB = dwHz / 2;

    const { timePoints, fid } = simulateFID(kAB, kBA, deltaA, deltaB, R20, R20);
    const { freqAxis, spectrum } = computeSpectrum(fid, timePoints);
    const specReal = spectrum.map(c => c.real);

    const maxSpec = Math.max(...specReal.filter(v => isFinite(v)));
    const yMax = Math.max(maxSpec * 1.15, 1);
    const freqRange = Math.max(dwHz * 0.8, 250);

    const specTraces = [
        {
            x: freqAxis, y: specReal,
            mode: 'lines',
            line: { color: '#7B2D3B', width: 2 },
            name: 'Spectrum',
            type: 'scatter'
        },
        {
            x: [deltaA, deltaA], y: [-yMax * 0.05, yMax],
            mode: 'lines',
            line: { color: '#A3445A', dash: 'dash', width: 1.5 },
            name: 'ω<sub>A</sub> (p<sub>A</sub>=' + pA.toFixed(2) + ')',
            type: 'scatter'
        },
        {
            x: [deltaB, deltaB], y: [-yMax * 0.05, yMax],
            mode: 'lines',
            line: { color: '#D4764E', dash: 'dash', width: 1.5 },
            name: 'ω<sub>B</sub> (p<sub>B</sub>=' + pB.toFixed(2) + ')',
            type: 'scatter'
        }
    ];

    const specLayout = {
        xaxis: { title: 'Frequency (Hz)', range: [-freqRange, freqRange] },
        yaxis: { title: 'Intensity', range: [-yMax * 0.05, yMax] },
        height: 350,
        margin: { l: 65, r: 20, t: 20, b: 55 },
        hovermode: 'closest',
        showlegend: true,
        legend: { x: 0.02, y: 0.98 }
    };

    Plotly.react('dispersion-spectrum-plot', specTraces, specLayout, { responsive: true, displayModeBar: true });
}

function initDispersion() {
    dispersionInitialized = true;
    generateDispNoise();

    // Slider input listeners
    ['disp-kex', 'disp-pB', 'disp-dw', 'disp-R20', 'disp-R20-800', 'disp-noise'].forEach(id => {
        document.getElementById(id).addEventListener('input', updateDispersionPlots);
    });

    // Checkbox listeners
    ['disp-dual-field', 'disp-show-fast', 'disp-ecpmg', 'disp-nucleus'].forEach(id => {
        document.getElementById(id).addEventListener('change', updateDispersionPlots);
    });

    // Resample noise button
    document.getElementById('disp-resample').addEventListener('click', () => {
        generateDispNoise();
        updateDispersionPlots();
    });

    // Preset buttons
    // dw values in ppm (previously Hz at 14.1T, now converted: Hz / 60.8 MHz)
    const presets = {
        'preset-fast':        { kex: 5000,  pB: 5,   dw: 1.64, R20: 10 },
        'preset-intermediate':{ kex: 1500,  pB: 5,   dw: 3.29, R20: 10 },
        'preset-slow':        { kex: 200,   pB: 5,   dw: 8.22, R20: 10 },
        'preset-no-exchange': { kex: 1000,  pB: 0.5, dw: 0.16, R20: 10 }
    };

    Object.entries(presets).forEach(([btnId, p]) => {
        document.getElementById(btnId).addEventListener('click', () => {
            document.getElementById('disp-kex').value = Math.log10(p.kex);
            document.getElementById('disp-pB').value = p.pB;
            document.getElementById('disp-dw').value = p.dw;
            document.getElementById('disp-R20').value = Math.log10(p.R20);
            generateDispNoise();
            updateDispersionPlots();
        });
    });

    // Initial render
    updateDispersionPlots();
}

// ============================================================
// Bootstrap
// ============================================================

// Wait for both DOM and Plotly to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { initTabs(); initBloch(); });
} else {
    initTabs();
    initBloch();
}

