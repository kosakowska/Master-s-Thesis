import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import statistics
import pandas as pd
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d


def read_from_csv_with_param_num_for_boxplot(file_path):  # modify for more than 3 parameter values
    list_ = []
    print(file_path)
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')

    filtered_list_of_lists = [[item for item in sublist if item != ''] for sublist in list_]

    list50 = []
    list100 = []
    list150 = []
    list200 = []
    list250 = []
    list300 = []
    list350 = []
    list400 = []
    list450 = []
    list500 = []

    for rows in filtered_list_of_lists:
        current_distance = int(rows[0])

        listFloat = [(float(current_distance) - float(element)) for element in rows[2:]]

        if current_distance == 50:
            list50.extend(listFloat)
        elif current_distance == 100:
            list100.extend(listFloat)
        elif current_distance == 150:
            list150.extend(listFloat)
        elif current_distance == 200:
            list200.extend(listFloat)
        elif current_distance == 250:
            list250.extend(listFloat)
        elif current_distance == 300:
            list300.extend(listFloat)
        elif current_distance == 350:
            list350.extend(listFloat)
        elif current_distance == 400:
            list400.extend(listFloat)
        elif current_distance == 450:
            list450.extend(listFloat)
        elif current_distance == 500:
            list500.extend(listFloat)

    return [list50, list100, list150, list200, list250, list300, list350, list400, list450, list500]


def read_from_csv_with_param_num(file_path):  # modify for more than 3 parameter values
    list_ = []
    print(file_path)
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')

    filtered_list_of_lists = [[item for item in sublist if item != ''] for sublist in list_]
    mean_list1 = []
    mean_list2 = []
    mean_list3 = []
    list1 = []
    list2 = []
    list3 = []

    for rows in filtered_list_of_lists:
        current_distance = rows[0]
        param_value = int(rows[1])

        listFloat = [(float(current_distance) - float(element)) for element in rows[2:]]

        if param_value == 1:
            list1.extend(listFloat)
            mean_list1.append(statistics.mean(listFloat))
        elif param_value == 2:
            list2.extend(listFloat)
            mean_list2.append(statistics.mean(listFloat))
        elif param_value == 3:
            list3.extend(listFloat)
            mean_list3.append(statistics.mean(listFloat))

    return list1, list2, list3, mean_list1, mean_list2, mean_list3


def read_from_csv_with_param_num1(filePath, num_params):
    lists = {f'list{i + 1}': [] for i in range(num_params)}
    mean_lists = {f'meanList{i + 1}': [] for i in range(num_params)}

    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                filtered_row = [item for item in row if item != '']
                current_distance = float(filtered_row[0])
                param_value = int(filtered_row[1])

                list_float = [current_distance - float(element) for element in filtered_row[2:]]

                list_key = f'list{param_value}'
                mean_list_key = f'meanList{param_value}'

                lists[list_key].extend(list_float)
                mean_lists[mean_list_key].append(statistics.mean(list_float))

                line_count += 1

    return tuple(lists.values()) + tuple(mean_lists.values())


def read_from_csv(file_path):
    list_ = []
    print(file_path)
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')

    filtered_list_of_lists = [[item for item in sublist if item != ''] for sublist in list_]
    new_list_after_deletion = []
    mean_list = []
    for rows in filtered_list_of_lists:
        current_distance = rows[0]
        temp_list = []
        for value in rows:
            if value != current_distance:
                new_value = float(value) - float(current_distance)
                if -350 <= new_value <= 350:
                    new_list_after_deletion.append(new_value)
                    temp_list.append(new_value)
        mean_list.append(statistics.mean(temp_list))
    return new_list_after_deletion, mean_list


def print_hist_with_dist(numbers, param_name, plot_number, plot_color):
    plt.figure(plot_number)
    plt.hist(numbers, bins=10, density=True, color=plot_color, alpha=0.5)
    mu, std = norm.fit(numbers)
    print("Mi: " + str(mu) + " STD: " + str(std))
    # xmin, xmax = plt.xlim()
    # plt.xticks(np.arange(-250, 350 + 1, 50))  # to have axis x with 50 step
    x = np.linspace(-250, 350, 110)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.xlabel('Błąd [cm]')
    plt.ylabel('Prawdopodobieństwo')
    plt.title('Histogram z krzywą rozkładu normalnego dla {}'.format(param_name))


def calculate_params_for_double_dist(numbers):
    # Compute KDE values directly
    kde_x = np.linspace(min(numbers), max(numbers), 1000)
    kde_y = sns.kdeplot(numbers).get_lines()[0].get_data()[1]  # Get KDE values
    kde_interp = interp1d(np.linspace(min(numbers), max(numbers), len(kde_y)), kde_y, kind='linear')
    kde_y = kde_interp(kde_x)

    # Define a function for a sum of two Gaussians
    def bimodal_gaussian(x, mu1, sigma1, mu2, sigma2, A1, A2):
        return A1 * norm.pdf(x, mu1, sigma1) + A2 * norm.pdf(x, mu2, sigma2)

    # Initial guess for the parameters
    # Compute basic statistics of the data
    mean = np.mean(numbers)
    std_dev = np.std(numbers)

    # Initial guess for mu1 and mu2: Choose values around the mean
    mu1_guess = mean - std_dev
    mu2_guess = mean + std_dev

    # Initial guess for sigma1 and sigma2: Choose a fraction of the standard deviation
    sigma1_guess = std_dev * 0.5
    sigma2_guess = std_dev * 0.5

    # Initial guess for A1 and A2: Choose based on the relative frequency of each peak
    # You can use the normalized histogram counts to estimate the relative frequencies
    hist, bin_edges = np.histogram(numbers, bins=20, density=True)
    peak1_freq = np.max(hist[:len(hist) // 2])  # Assuming first peak is on the left half
    peak2_freq = np.max(hist[len(hist) // 2:])  # Assuming second peak is on the right half
    total_freq = np.sum(hist)
    A1_guess = peak1_freq / total_freq
    A2_guess = peak2_freq / total_freq
    # Combine initial guesses into a list
    initial_guess = [mu1_guess, sigma1_guess, mu2_guess, sigma2_guess, A1_guess, A2_guess]

    # Fit the data to the bimodal Gaussian function
    params, cov = curve_fit(bimodal_gaussian, kde_x, kde_y, p0=initial_guess)

    # Plot the fitted curve
    # plt.plot(kde_x, bimodal_gaussian(kde_x, *params), color='red', linestyle='--', label='Fitted Curve')
    # Extract fitted parameters
    mu1, sigma1, mu2, sigma2, A1, A2 = params
    print("Fitted Parameters (mu1, sigma1, mu2, sigma2, A1, A2):", mu1, sigma1, mu2, sigma2, A1, A2)


def print_hist_with_double_dist(numbers, param_name, plot_number, plot_color):
    plt.figure(plot_number)
    plt.xlim(-300, 200)
    # Plot histogram with KDE using seaborn
    sns.histplot(numbers, stat='probability', bins=15, kde=True, color=plot_color, alpha=0.5)
    # plt.hist(numbers, bins=20, color='blue', alpha=0.5, edgecolor='black')
    plt.title('Histogram z krzywą rozkładu dwumodalnego dla {}'.format(param_name))
    plt.xlabel('Błąd [cm]')
    plt.ylabel('Prawdopodobieństwo')

    calculate_params_for_double_dist(numbers)
    plt.show()


def print_cdf(data, param_name, name):
    count, bins_count = np.histogram(data, bins=50)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label=name)
    plt.title('Dystrybuanta rozkładu dla {}'.format(param_name))
    plt.ylabel('Prawdopodobieństwo')
    plt.xlabel('Błąd [cm]')
    plt.legend()


def print_box_plot(data, param_name, plot_number, distance=False):
    plt.figure(plot_number)
    ax = sns.boxplot(data=data)
    if distance:
        ax.set_xticklabels(np.linspace(50, 500, 10))
        plt.title('Wykres pudełkowy dla parametru {}'.format(param_name))
        plt.xlabel('Odległość [cm]')
    else:
        plt.title('Wykres pudełkowy dla parametru {}'.format(param_name))
        plt.xlabel('Wartość parametru')
    plt.ylabel('Błąd [cm]')


def is_inside(condition):
    return "wewnątrz" if condition else "na zewnątrz"


def print_bar_graph(data, title, param_name, is_inside_, plot_number, need_two_bars=False, data2=None, param_name2=None,
                    need_three_bars=False, data3=None, param_name3=None,
                    plotColor='blue',
                    plotColor2='green',
                    plotColor3='magenta'):
    if data2 is None:
        data2 = []
    if data3 is None:
        data3 = []
    inOrOut = is_inside(is_inside_)
    plt.figure(plot_number)

    distances = np.linspace(50, 500, 10)
    plt.bar(distances, data, label=param_name, color=plotColor, width=12)
    bar_width = 16
    bar_positions_series2 = distances + bar_width
    bar_positions_series3 = distances + 2 * bar_width

    if need_two_bars:
        plt.bar(bar_positions_series2, data2, label=param_name2, color=plotColor2, width=12)
    if need_three_bars:
        plt.bar(bar_positions_series3, data3, label=param_name3, color=plotColor3, width=12)

    plt.xticks(np.arange(50, 500 + 1, 50))  # to have axis x with 50 step
    plt.title('Wykres słupkowy dla parametru {} {}'.format(title, inOrOut))
    plt.xlabel('Odległość [cm]')
    plt.ylabel('Błąd [cm]')
    plt.legend()


if __name__ == '__main__':
    # outside
    numbers_default_out, mean_numbers_default_out = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\default\default.csv')
    numbers_lci_out, mean_numbers_lci_out = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\lci\lci.csv')
    numbers_civic_out, mean_numbers_civic_out = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\civic\civic.csv')
    numbers_retries_out1, numbers_retries_out2, numbers_retries_out3, mean_numbers_retries_out1, mean_numbers_retries_out2, mean_numbers_retries_out3 = read_from_csv_with_param_num(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100'
        r'\retries\retries.csv')
    burst_period_out = read_from_csv_with_param_num1(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\burst_period'
        r'\burst_period.csv', 31)
    burst_duration_out = read_from_csv_with_param_num1(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\burst_duration'
        r'\burst_duration.csv', 15)
    spb_out = read_from_csv_with_param_num1(r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\spb\spb.csv', 31)
    spb_in = read_from_csv_with_param_num1(r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\spb\spb.csv', 31)

    burst_period_out_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\burst_period'
        r'\burst_period.csv')
    burst_duration_out_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\burst_duration\burst_duration.csv')
    spb_out_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOgród100\spb\spb.csv')
    spb_in_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\spb\spb.csv')

    # inside
    numbers_civic_in, mean_numbers_civic_in = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\civic\civic.csv')

    numbers_lci_in, mean_numbers_lci_in = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\lci\lci.csv')
    numbers_default_in, mean_numbers_default_in = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\default'
        r'\default.csv')
    numbers_retries_in1, numbers_retries_in2, numbers_retries_in3, mean_numbers_retries_in1, mean_numbers_retries_in2, mean_numbers_retries_in3 = read_from_csv_with_param_num(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100'
        r'\retries\retries.csv')

    burst_period_in = read_from_csv_with_param_num1(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\burst_period\burst_period.csv', 31)
    burst_duration_in = read_from_csv_with_param_num1(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\burst_period\burst_period.csv', 31)

    burst_period_in_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\burst_period\burst_period.csv')
    burst_duration_in_box = read_from_csv_with_param_num_for_boxplot(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryDom100\burst_duration\burst_duration.csv')

    numbers_default_tcp, mean_numbers_default_tcp = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\TCP\default.csv')
    numbers_default_udp, mean_numbers_default_udp = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\UDP\default.csv')

    numbers_default_tcpOut, mean_numbers_default_tcpOut = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\TCPOutside\default.csv')
    numbers_default_udpOut, mean_numbers_default_udpOut = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\UDPOutside\default.csv')

    numbers_optimal_inside, mean_numbers_optimal_inside = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOptimalIn\default.csv')
    numbers_optimal_outside, mean_numbers_optimal_outside = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\PomiaryOptimalOut\default.csv')

    numbers_default_out_new, mean_numbers_default_out_new = read_from_csv(
        r'C:\Users\kosak\OneDrive\Pulpit\Magisterka\OutDefault\default.csv')
