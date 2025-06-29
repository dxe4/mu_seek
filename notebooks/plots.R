library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)

script_path <- dirname(normalizePath(sys.frame(1)$ofile))
file_path <- file.path(script_path, "../data/processed/", "bpm_and_sound_neutrality.parquet")
df  <- read_parquet(file_path)


custom_shapes <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

df <- df %>%
  separate(style_genre, into = c("genre", "style"), sep = "\\|") %>%
  mutate(
    genre = str_trim(genre),
    style = str_trim(style)
  )

df$genre <- factor(df$genre)
# Plot
p <- ggplot(df, aes(
  x = estimated_bpm,
  y = style,
  color = estimated_natural_sound_pct,
  # shape = genre
)) +
  geom_point(size = 1.4) +
  labs(
    title = "Estimated BPM vs Style by Genre",
    x = "Estimated BPM",
    y = "Style",
    color = "Estimated Natural Sound (%)",
    # shape = "Genre"
  ) +
  # scale_shape_manual(values = custom_shapes) +
  scale_color_gradient(low = "blue", high = "red") +
  facet_wrap(~ genre, ncol = 7) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "right",
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 10),
    plot.title = element_text(size = 18),
    strip.text = element_text(size = 8)
  )


  ggsave(file.path(script_path, "../plots/scatter_plot_large.png"),
         plot = p,
         width = 9000,
         height = 2400,
         units = "px",
         dpi = 300)

  p

  pdf(file.path(script_path, "../plots/scatter_plot_large.pdf"), width = 10, height = 300)
  print(p)
  dev.off()
