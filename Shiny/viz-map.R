library(shiny)
library(plotly)
library(dplyr)

# Load the Philadelphia data
philly_data <- read.csv("data_philly.csv", stringsAsFactors = FALSE)

# Clean and prepare data
philly_data$Year <- as.numeric(philly_data$Year)
philly_data <- philly_data[!is.na(philly_data$Year) & !is.na(philly_data$lat) & !is.na(philly_data$lon), ]

# Get unique types and years for filters
venue_types <- c("All", sort(unique(unlist(strsplit(philly_data$type, ",")))))
venue_types <- venue_types[venue_types != "" & !is.na(venue_types)]
years <- sort(unique(philly_data$Year))

ui <- fluidPage(
  titlePanel("Philadelphia LGBTQ Venues - Interactive Map"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("type", "Venue Type:", 
                  choices = venue_types, 
                  selected = "All"),
      sliderInput("year_range", "Year Range:",
                  min = min(years), 
                  max = max(years),
                  value = c(min(years), max(years)),
                  step = 1,
                  sep = ""),
      hr(),
      h4("Map Statistics:"),
      textOutput("venue_count")
    ),
    
    mainPanel(
      plotlyOutput("map", height = "600px"),
      hr(),
      h4("Selected Venues:"),
      dataTableOutput("venue_table")
    )
  )
)

server <- function(input, output) {
  
  # Filter data based on selections
  filtered_data <- reactive({
    df <- philly_data
    
    # Filter by type
    if (input$type != "All") {
      df <- df[grepl(input$type, df$type), ]
    }
    
    # Filter by year range
    df <- df[df$Year >= input$year_range[1] & df$Year <= input$year_range[2], ]
    
    return(df)
  })
  
  # Create the map
  output$map <- renderPlotly({
    df <- filtered_data()
    
    # Create hover text
    df$hover_text <- paste0(
      "<b>", df$title, "</b><br>",
      "Address: ", df$streetaddress, "<br>",
      "Type: ", df$type, "<br>",
      "Year: ", df$Year
    )
    
    # Create the plotly map
    plot_ly(df, 
            lat = ~lat, 
            lon = ~lon,
            type = 'scattermapbox',
            mode = 'markers',
            marker = list(size = 10, color = ~Year, 
                         colorscale = 'Viridis',
                         showscale = TRUE,
                         colorbar = list(title = "Year")),
            text = ~hover_text,
            hoverinfo = 'text') %>%
      layout(
        mapbox = list(
          style = "open-street-map",
          zoom = 11,
          center = list(lon = mean(df$lon, na.rm = TRUE), 
                       lat = mean(df$lat, na.rm = TRUE))
        ),
        title = paste("Philadelphia LGBTQ Venues:", 
                     input$year_range[1], "-", input$year_range[2])
      )
  })
  
  # Display venue count
  output$venue_count <- renderText({
    paste("Total venues shown:", nrow(filtered_data()))
  })
  
  # Display data table
  output$venue_table <- renderDataTable({
    df <- filtered_data()
    df %>%
      select(title, streetaddress, type, Year) %>%
      arrange(Year, title)
  })
}

shinyApp(ui = ui, server = server)