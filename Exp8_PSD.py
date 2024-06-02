import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks

# Load the dataset
isotope = sys.argv[1]
raw_data = np.loadtxt(fr"C:\Users\jerem\OneDrive\Grad School\NUCE 597\Exp8\{isotope}_001_wf_0.dat")
w = 4096  # 12-bit digitizer
rows, columns = raw_data.shape
two_d = np.zeros((rows // 96, 96))

# Parsing the raw data and changing polarity
counter1 = 0
for i in range(rows // 96):
    for j in range(96):
        two_d[i, j] = w - raw_data[counter1, 1]
        counter1 += 1

# Troubleshooting
average_two_d = np.mean(two_d, axis=0)
fig = px.line(y=average_two_d, title=f'{isotope} Average Channel vs. Sample Number', labels={'index': 'Sample Number (4ns each)', 'y': 'Average Channel'})
fig.show()

max_two_d = np.max(two_d, axis=1)

# Computing all of the ratios
a, b, c = 7, 20, 96
chop = np.zeros(rows // 96)
head = np.zeros(rows // 96)
tail = np.zeros(rows // 96)
TTF = np.zeros(rows // 96)
TTH = np.zeros(rows // 96)
TTM = np.zeros(rows // 96)
full = np.zeros(rows // 96)

for i in range(rows // 96):
    chop[i] = np.mean(two_d[i, :a])
    head[i] = np.sum(two_d[i, a:b] - chop[i])
    tail[i] = np.sum(two_d[i, b:] - chop[i])
    full[i] = head[i] + tail[i]
    TTF[i] = tail[i] / full[i]
    TTH[i] = tail[i] / head[i]
    TTM[i] = tail[i] / max_two_d[i]

# Function to compute axis limits by removing significant outliers
def compute_axis_limits(data, factor=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return np.min(filtered_data), np.max(filtered_data)

# Compute axis limits
x_min_full, x_max_full = compute_axis_limits(full)
y_min_tail, y_max_tail = compute_axis_limits(tail)
x_min_max_two_d, x_max_max_two_d = compute_axis_limits(max_two_d)
y_min_TTF, y_max_TTF = compute_axis_limits(TTF, factor=7)  # TTF values typically require a larger factor
y_min_TTH, y_max_TTH = compute_axis_limits(TTH, factor=7)
y_min_TTM, y_max_TTM = compute_axis_limits(TTM, factor=7)

# Plotting histograms and scatter plots
fig = px.histogram(TTF, nbins=10000, title=f'{isotope} Histogram of Tail to Total Ratios', labels={'value': 'Tail to Total Ratio (10000 bins)', 'count': 'Counts per Bin'})
fig.update_xaxes(range=[y_min_TTF, y_max_TTF])
fig.show()

fig = px.scatter(x=full, y=tail, title=f'{isotope} Tail vs. Total Integral', labels={'x': 'Total Integral', 'y': 'Tail Integral'})
fig.update_xaxes(range=[x_min_full, x_max_full])
fig.update_yaxes(range=[y_min_tail, y_max_tail])
fig.show()

fig = px.scatter(x=max_two_d, y=TTF, title=f'{isotope} Tail to Total Ratio vs. Max Pulse Height', labels={'x': 'Max Pulse Height', 'y': 'Tail to Total Ratio'})
fig.update_xaxes(range=[x_min_max_two_d, x_max_max_two_d])
fig.update_yaxes(range=[y_min_TTF, y_max_TTF])
fig.show()

fig = px.histogram(TTH, nbins=10000, title=f'{isotope} Histogram of Tail to Head Ratios', labels={'value': 'Tail to Head Ratio (10000 bins)', 'count': 'Counts per Bin'})
fig.update_xaxes(range=[y_min_TTH, y_max_TTH])
fig.show()

fig = px.scatter(x=head, y=tail, title=f'{isotope} Tail vs. Head Integral', labels={'x': 'Head Integral', 'y': 'Tail Integral'})
fig.update_xaxes(range=[x_min_full, x_max_full])
fig.update_yaxes(range=[y_min_tail, y_max_tail])
fig.show()

fig = px.scatter(x=max_two_d, y=TTH, title=f'{isotope} Tail to Head Ratio vs. Max Pulse Height', labels={'x': 'Max Pulse Height', 'y': 'Tail to Head Ratio'})
fig.update_xaxes(range=[x_min_max_two_d, x_max_max_two_d])
fig.update_yaxes(range=[y_min_TTH, y_max_TTH])
fig.show()

fig = px.histogram(TTM, nbins=10000, title=f'{isotope} Histogram of Tail to Max Pulse Height Ratios', labels={'value': 'Tail to Max Pulse Height Ratio (10000 bins)', 'count': 'Counts per Bin'})
fig.update_xaxes(range=[y_min_TTM, y_max_TTM])
fig.show()

fig = px.scatter(x=max_two_d, y=tail, title=f'{isotope} Tail Integral vs. Max Pulse Height', labels={'x': 'Max Pulse Height', 'y': 'Tail Integral'})
fig.update_xaxes(range=[x_min_max_two_d, x_max_max_two_d])
fig.update_yaxes(range=[y_min_tail, y_max_tail])
fig.show()

fig = px.scatter(x=max_two_d, y=TTM, title=f'{isotope} Tail to Max Pulse Height Ratio vs. Max Pulse Height', labels={'x': 'Max Pulse Height', 'y': 'Tail to Max Pulse Height Ratio'})
fig.update_xaxes(range=[x_min_max_two_d, x_max_max_two_d])
fig.update_yaxes(range=[y_min_TTM, y_max_TTM])
fig.show()

# Counting neutrons vs. gammas based on cutoffs
neutrons_TTF = np.sum(TTF > 0.13)
gammas_TTF = np.sum(TTF <= 0.13)

neutrons_TTH = np.sum(TTH > 0.13)
gammas_TTH = np.sum(TTH <= 0.13)

neutrons_TTM = np.sum(TTM > 0.44)
gammas_TTM = np.sum(TTM <= 0.44)

print(f'Neutrons (TTF): {neutrons_TTF}, Gammas (TTF): {gammas_TTF}')
print(f'Neutrons (TTH): {neutrons_TTH}, Gammas (TTH): {gammas_TTH}')
print(f'Neutrons (TTM): {neutrons_TTM}, Gammas (TTM): {gammas_TTM}')

# Function to calculate the Figure of Merit (FoM)
def calculate_fom(parameter):
    counts, bin_edges = np.histogram(parameter, bins=1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks
    peaks, _ = find_peaks(counts, height=100)  # Adjust height as needed
    if len(peaks) < 2:
        raise ValueError("Not enough peaks found to calculate FoM.")

    neutron_peak = bin_centers[peaks[0]]
    gamma_peak = bin_centers[peaks[1]]

    # Calculate FWHM for each peak
    def fwhm(data, peak_index):
        half_max = data[peak_index] / 2
        left_index = np.where(data[:peak_index] <= half_max)[0][-1]
        right_index = np.where(data[peak_index:] <= half_max)[0][0] + peak_index
        return bin_centers[right_index] - bin_centers[left_index]

    fwhm_neutron = fwhm(counts, peaks[0])
    fwhm_gamma = fwhm(counts, peaks[1])

    # Calculate separation
    separation = gamma_peak - neutron_peak

    # Calculate FoM
    FoM = separation / (fwhm_neutron + fwhm_gamma)

    return FoM

# Calculate FoMs for each parameter
fom_ttf = calculate_fom(TTF)
fom_tth = calculate_fom(TTH)
fom_ttm = calculate_fom(TTM)

# Print FoMs in a formatted table
print("\nTable 2 - Figure of Merit (FOM) Comparison")
print("{:<40} {:<10}".format("Method", "FOM"))
print("{:<40} {:<10}".format("(1) Tail Integral vs. Full Integral", fom_ttf))
print("{:<40} {:<10}".format("(2) Tail Integral vs. Head Integral", fom_tth))
print("{:<40} {:<10}".format("(3) Tail Integral vs. Maximum Pulse-Height", fom_ttm))

