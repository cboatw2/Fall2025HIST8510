library(shiny)
library(ggplot2)

# Load the data
philly_data <- read.csv("data_philly.csv", stringsAsFactors = FALSE)

# Clean and prepare data
philly_data$Year <- as.numeric(philly_data$Year)
philly_data <- philly_data[!is.na(philly_data$Year), ]

# Get unique types for dropdown
venue_types <- c("All", sort(unique(unlist(strsplit(philly_data$type, ",")))))
venue_types <- venue_types[venue_types != "" & !is.na(venue_types)]

ui <- fluidPage(
  titlePanel("Philadelphia Venues by Type and Year"),
  sidebarLayout(
    sidebarPanel(
      selectInput("type", "Venue Type:", choices = venue_types, selected = "All")
    ),
    mainPanel(
      plotOutput("barPlot")
    )
  )
)

server <- function(input, output) {
  filtered_data <- reactive({
    if (input$type == "All") {
      philly_data
    } else {
      philly_data[grepl(input$type, philly_data$type), ]
    }
  })
  
  output$barPlot <- renderPlot({
    df <- filtered_data()
    ggplot(df, aes(x = Year, fill = type)) +
      geom_bar() +
      labs(title = paste("Number of Venues by Year", ifelse(input$type == "All", "", paste("(", input$type, ")"))),
           x = "Year", y = "Number of Venues") +
      theme_minimal()
  })
}

shinyApp(ui = ui, server = server)